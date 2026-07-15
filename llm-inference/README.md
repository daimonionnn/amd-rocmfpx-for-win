# ROCmFPX for Windows + headless LLM serving — [Not Only] For AMD Strix Halo

The core of this project, in order:

1. **ROCmFPX support for Windows** — a native Windows build and runtime for the
   [ROCmFPX](https://github.com/charlie12345/ROCmFPX) llama.cpp fork (AMD-specific GGUF weight
   formats). Upstream ships no releases and only bash/Linux build scripts; `Setup-ROCmFPX.ps1`
   is the Windows port (details and evaluation in §8), `Get-ROCmFPXModel.ps1` fetches models.
2. **Headless llama-server** — `Serve-Qwen.ps1`: an OpenAI-compatible, LAN-reachable API server
   (+ chat WebUI on the same port), switchable between the production ROCm 7 runtime and the
   ROCmFPX fork. Serves the real workload: a **Hermes agent running 100K+ token contexts** on
   **Qwen3.6-27B**, quality-first **Q8**.
3. **Measurement-driven optimization findings** for long-context inference on this box —
   the numbered sections below (prefill wall, ROCm 7 build, MTP, quant trade-offs, ROCmFPX).

**Windows-first.** Everything is native PowerShell on Windows 11, tested and optimized for AMD
Strix Halo. Most comparable tooling (vLLM/SGLang serving stacks, ROCm build scripts) is
Linux-first; Windows options for AMD APU serving are thin on the ground — that gap is what this
project fills.

- Machine: Minisforum MS-S1 MAX — AMD Ryzen AI MAX+ 395 / Radeon 8060S (**gfx1151**),
  128 GB LPDDR5X @ 8000 MT/s, BIOS 1.08, IOMMU disabled
- Production runtime: ROCm 7 gfx1151 llama.cpp build in `..\llm-bench\bin\`
- Sibling add-on: `..\llm-bench` — llama-bench benchmark harness

## Why this is separate from `llm-bench`

`llm-bench` measures short prefill (`pp512`) + generation (`tg128`). Those are **not**
the bottleneck for a 100K-token agent — at that length, attention goes O(S²) and TTFT is
dominated by long-context prefill and the KV cache, not the short-prompt GEMM path.

## Findings so far

### 1. Short-prefill tuning is a dead end (measured)

`scripts\tune-prefill.ps1`, Gemma 27B dense, prefill-only:

| Config                    | pp512 | pp2048          |
|---------------------------|------:|----------------:|
| build default             |   415 | 378             |
| `ROCBLAS_USE_HIPBLASLT=1` |   408 | 369             |
| `ROCBLAS_USE_HIPBLASLT=0` |   413 | 368             |
| `-ub 256 / 512 / 1024`    |     — | 361 / 365 / 371 |

Forcing hipBLASLt does nothing (b9910 already dispatches to tuned gfx1151 kernels);
`-ub 1024` buys ~+3% at most. Qwen MoE confirms the same. **No knob left on short prefill.**

### 2. The problem, measured — vanilla long-context prefill (Qwen 27B Q4_K_XL, b9910)

`scripts\longctx-prefill.ps1`, `-fa on -ub 1024`, prefill-only. TTFT = tokens ÷ rate:

| Context | Prefill t/s | TTFT        |
|--------:|------------:|------------:|
| 4,096   |       325.6 | ~13 s       |
| 16,384  |       219.7 | ~75 s       |
| 32,768  |       127.9 | ~4.3 min    |
| 65,536  |        67.9 | ~16 min     |
| 131,072 |        32.3 | **~68 min** |

Throughput ~halves per context doubling (O(S²) attention wall). A 128K prefill on vanilla
llama.cpp is **~68 minutes** of TTFT on Q4 — and the user's target **Q8 is slower still**.
This is the number every optimization below must beat. Source: `results\longctx-prefill.csv`.

### 3. Real levers for long context (in priority order)

1. **ROCm 7 gfx1151-specific build — BIG WIN, measured, ZERO quality loss. Do this first.**
   `lemonade-sdk/llamacpp-rocm` b1295 (ROCm 7, compiled for gfx1151) vs official multi-arch
   b9910, same settings (`-fa on -ub 1024`), Qwen 27B prefill:

   | Context | b9910 t/s | ROCm 7 t/s | Speedup   | ROCm 7 TTFT                  |
   |--------:|----------:|-----------:|:---------:|-----------------------------:|
   | 4,096   |     325.6 |      356.1 | +9%       | ~11 s                        |
   | 16,384  |     219.7 |      311.7 | +42%      | ~53 s                        |
   | 32,768  |     127.9 |      271.3 | 2.1×      | ~2 min                       |
   | 65,536  |      67.9 |      214.5 | 3.2×      | ~5 min                       |
   | 131,072 |      32.3 |  **150.6** | **4.66×** | **~14.5 min** (from ~68 min) |

   Speedup GROWS with context (flatter curve = much better long-context FA kernels). Unlike
   PFlash this is **exact** — no fidelity trade. **This should be the new 1 binary.**
   Build dir: `..\llm-bench\bin-rocm7-gfx1151\`. Source: `results\longctx-prefill-rocm7.csv`.
   Note: on ROCm 7 the `rocWMMA` FA flag is reportedly slower at depth on gfx1151 — stay on the
   default HIP path (`-fa on`), which is what these numbers use.
2. **Q8 KV cache** (`-ctk q8_0 -ctv q8_0`) — **MEASURED: skip it.** On the ROCm 7 build, Q8 KV
   gives **zero prefill speedup** vs f16 KV (prefill is compute-bound, not KV-bound): 16K 320.3
   vs 320.6, 32K 280.1 vs 277.7, 65K 223.1 vs 220.7 — identical. And Q8 KV **crashed at 128K**
   (f16 KV succeeded there) — likely a FA + quantized-V kernel bug. Its only benefit is memory,
   which the user does not need (speed > memory, see [[strix-halo-workload]]). **Use f16 KV.**

   Also measured: the **Q8_0 *model*** prefills at the SAME speed as Q4 on ROCm 7 (128K = 152.7
   vs 150.6 t/s) — prefill doesn't care about weight size. The Q8 penalty is purely in *decode*
   (memory-bound); see `results\decode-quant-compare.csv`.
3. **Lucebox PFlash** — purpose-built long-context prefill, ~3× at 16K / headline 10.4× at very
   long ctx (O(S²)→O(S)). **Lossy**: keeps ~5% of tokens via a Qwen3-0.6B drafter. Validated on
   needle-retrieval, NOT agentic tool-recall — **must validate on real Hermes traces** before
   trusting. `ghcr.io/luce-org/lucebox-hub:rocm`. Consider only if ROCm 7 + Q8-KV isn't enough
   and the accuracy check passes.

### 4. Decode speed is purely memory-bandwidth bound (measured)

`scripts\decode-quant-compare.ps1`, Qwen3.6-27B, ROCm 7 build, `tg128`:

| Quant  | Size     | tg128 (fresh) | tg128 @ 32K | t/s × GiB |
|--------|---------:|--------------:|------------:|----------:|
| Q4_K_M | 15.4 GiB |         12.73 |       11.32 |       196 |
| Q6_K   | 20.6 GiB |          9.55 |        8.72 |       196 |
| Q8_0   | 26.6 GiB |          7.60 |        7.14 |       202 |

`t/s × model-size ≈ constant (~198)` ⇒ decode is **purely memory-bandwidth bound** — effective
~210 GB/s ≈ **83% of the 256 GB/s LPDDR5X ceiling**. This is the "bursty GPU (100%→0%)" the user
saw: memory starvation, NOT thermal/clock throttling.

Implications (speed-first): **Q8 decode is ~40% slower than Q4** (Q6 −25%), but **prefill is quant-
independent** (see §3). The real decode unlock is **speculative decoding (MTP)** — processes multiple
tokens per weight-stream, bypassing the bandwidth wall (~12.7 → ~25 t/s). Best combo for Q8 quality +
speed: **Q8 + MTP** (`unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q8_K_XL.gguf`).

### 5. Q8 + MTP on long context (the recommended config) — measured

`scripts\longctx-q8-mtp.ps1`, Qwen3.6-27B-**Q8_K_XL** (33.3 GB, unsloth MTP pack), ROCm 7 build,
llama-cli, natural-prose prompt. Numbers from llama.cpp's own `[ Prompt | Generation ]` summary:

| Context | Mode  | Prefill t/s | Decode t/s | MTP decode speedup |
|--------:|-------|------------:|-----------:|:------------------:|
| ~34.5K  | plain |       279.0 |        6.2 | —                  |
| ~34.5K  | +MTP  |       266.5 |   **13.1** | **2.11×**          |
| ~137K   | plain |       153.9 |        5.4 | —                  |
| ~137K   | +MTP  |       145.0 |    **9.9** | **1.83×**          |

MTP ~doubles decode (2.11× @32K, 1.83× @128K) and barely touches prefill (−6%, draft overhead).
Plain decode here is lower than Q8_0 earlier because Q8_K_XL is bigger (33.3 vs 26.6 GB) — matches
the bandwidth rule (198/33.3 ≈ 5.9). Prompt is natural prose; real agent content (code/structured
tool output) is more predictable, so real MTP acceptance/speedup is likely **higher**.

## FINAL recommended config (137K-context Hermes agent)

|                | Old (b9910, Q8) | **Recommended (ROCm 7 + Q8_K_XL + MTP)** |
|----------------|-----------------|------------------------------------------|
| TTFT (prefill) | ~68 min         | **~15.5 min** (4.4× faster)              |
| Decode         | ~5–6 t/s        | **~9.9 t/s** (1.8× via MTP)              |
| KV cache       | —               | f16 (Q8-KV gives nothing here)           |
| Quality        | Q8              | Q8, no loss (unlike PFlash)              |

Model: `unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-UD-Q8_K_XL.gguf`; binary: `..\llm-bench\bin\`
(ROCm 7); flags: `-fa on --spec-type draft-mtp --spec-draft-n-max 4 -ngl -1`.

Serving default is shipped via `.\Serve-Qwen.ps1`, which uses **Q8_0** (26.6 GB) — same
quant LM Studio runs, near-identical quality to Q8_K_XL but ~1.27× faster decode (§6). Override
with `-Model ...Q8_K_XL.gguf` for max quality or `...Q4_K_XL.gguf` for max speed.

### 7. Q4 vs Q8 decode at long context — the quant advantage VANISHES (measured)

`scripts\longctx-q4-mtp.ps1` (Q4_K_XL) vs §5 (Q8_K_XL), both +MTP, same prompts/method:

| Context | Metric  | Q4_K_XL   | Q8_K_XL |
|--------:|---------|----------:|--------:|
| 32K     | prefill |     259.2 |   266.5 |
| 32K     | decode  | **17.50** |    13.1 |
| 128K    | prefill |     143.4 |   145.0 |
| 128K    | decode  |  **9.90** | **9.9** |

Prefill is quant-independent (as always). Decode: Q4 is ~1.34× faster at 32K, but at **128K Q4 = Q8
(both 9.9 t/s)** — the advantage disappears. Why: at long context the **f16 KV cache (~35 GB at 128K)
dominates memory traffic**, and KV is the same size regardless of weight quant. So weights stop being
the bottleneck. **Practical:** for the 100K+ agent workload, Q4 buys almost nothing over Q8 on decode
→ stay on Q8 for quality. Q4's decode edge only shows at shorter contexts (≤~32K).

### 8. ROCmFPX — AMD-specific weight formats (measured: speed §below, quality −1.7% PPL)

[ROCmFPX](https://github.com/charlie12345/ROCmFPX) is a llama.cpp fork adding AMD-only GGUF weight
formats — `Q4_0_ROCMFP4` (4.25–4.50 bpw), `Q6_0_ROCMFPX`, `Q8_0_ROCMFPX`, and `_AGENT` presets that
claim to protect coherency on structured output (JSON, tool-calling, code). These are *model-weight*
formats, not KV-cache tricks, and **stock llama.cpp cannot load them** — they need the fork's runner.

Built for Windows/gfx1151 via `.\Setup-ROCmFPX.ps1` → `bin-rocmfpx\` (one build, both
`-dev ROCm0` and `-dev Vulkan0`; upstream claims Vulkan is the stronger decode path on Strix
Halo). Get a model with `.\Get-ROCmFPXModel.ps1` → `models\`; run with
`.\Serve-Qwen.ps1 -Runtime rocmfpx`. Build details (requirements, the MSVC 14.44 pin and why)
are documented in `Setup-ROCmFPX.ps1` itself.

**Measured (full quick sweep, `-r 3`, clean GPU, IOMMU off / BIOS 1.08, no MTP).**
Source: `results\rocmfpx-ab-iommu-off.csv`; the pre-BIOS-change baseline is
`results\rocmfpx-ab-iommu-on.md`.

| Config                               | Device  | pp4096 | pp16384 | pp32768 | tg128     | tg128 @32K |
|--------------------------------------|---------|-------:|--------:|--------:|----------:|-----------:|
| Q8_0 (27.04 GiB), production `bin\`  | ROCm0   |  364.5 |   305.3 |   271.1 |      7.66 |       7.19 |
| Q8_0 (27.04 GiB), fork               | ROCm0   |  369.8 |   327.0 |   285.6 |      7.68 |       7.23 |
| Q4_0_ROCMFP4_STRIX (15.69 GiB), fork | ROCm0   |  367.4 |   325.7 |   283.7 | **13.74** |  **12.20** |
| Q4_0_ROCMFP4_STRIX (15.69 GiB), fork | Vulkan0 |  292.1 |   252.7 |   181.1 |     14.32 |      12.56 |

What falls out of this:

1. **The fork is not a regression on standard quants.** Q8_0 on the fork matches (even slightly
   beats) the production ROCm 7 build on both prefill and decode. Switching runtimes costs
   nothing, so the only question is whether the ROCmFPX *formats* buy anything.
2. **Prefill is quant-independent, as §3 predicted.** ROCmFP4 prefills at the same speed as Q8_0
   on the same runtime (367/326/284 vs 370/327/286) despite being 42% smaller. No prefill lever
   here.
3. **ROCmFP4's decode win is just the bandwidth rule, not better kernels.** 13.74 vs 7.68 t/s =
   1.79×, and the weight-size ratio is 27.04/15.69 = 1.72×. `t/s × GiB` ≈ 216 vs the ~196–208
   bandwidth line of §4 — a few percent of kernel edge at most. It is a smaller model, not a
   faster format.
4. **Vulkan is the wrong device for this box.** Upstream's "Vulkan is the stronger decode path"
   holds, but barely (+4% decode) — and it costs **−36% on prefill at 32K** (181 vs 284). Prefill
   is our entire bottleneck (§3), so **stay on `-dev ROCm0`.**

**Quality, measured (`scripts\rocmfpx-ppl.ps1`, full wikitext-2 test set, same runner):**

| Model                          | Wikitext-2 PPL    | vs Q8_0         |
|--------------------------------|-------------------|-----------------|
| Q8_0 (27.04 GiB)               | **6.906 ± 0.045** | — (reference)   |
| Q4_0_ROCMFP4_STRIX (15.69 GiB) | 7.022 ± 0.046     | **+1.7% worse** |

The gap is real (≈2.5× the standard error) but small — typical Q4_K_M sits at +2–4% over Q8 at
this model size, so the imatrix FP4 recipe holds up well. Caveat: wiki-text perplexity does not
measure tool-calling discipline or long-context recall, which is where 4-bit errors usually hurt
agents most. Verdict unchanged: Q8 for the 100K+ agent (where FP4 buys no decode anyway, see
below), FP4 for short interactive use where its 1.8× decode is actually felt.

**The 128K points — measured (`scripts\rocmfpx-128k.ps1`, fork runtime, ROCm0, true decode at
depth via `-d 131072`, IOMMU off):**

| Config @128K       | pp131072 | tg128 @d131072 |
|--------------------|---------:|---------------:|
| ROCmFP4 (15.7 GiB) |    155.9 |       **9.19** |
| Q8_0 (27.0 GiB)    |    156.0 |           6.06 |

Three findings: (1) **prefill is perfectly quant-independent at 128K** (155.9 ≈ 156.0), and the
fork keeps a small prefill edge over lemonade even here (156 vs 152). (2) **FP4's decode edge
narrows but does NOT vanish: 1.79× short → 1.52× @128K.** §7's full Q4=Q8 convergence (both
9.9 t/s) was measured *with MTP* — in raw decode the KV traffic only eats part of the weight
advantage. (3) The intriguing corollary: FP4's raw 9.19 t/s is already near the production
Q8+MTP 9.9 t/s — **FP4+MTP could plausibly be the fastest 128K decode on this box** (unmeasured;
the fork's MTP path is slower per §above, so not a given). It would still cost −1.7% PPL →
**the production Q8_0 config in §5 stands on quality grounds**, but FP4+MTP is now a legitimate
speed-first option to measure rather than a ruled-out one.

**Side finding — BIOS IOMMU off (+ BIOS 1.06→1.08) helps prefill only, and grows with context.**
Decode was completely flat (7.57→7.66 fresh, 12.21→12.20 fp4 @32K — bandwidth unaffected), but
ROCm-path prefill gained +1% at 4K, +6% at 16K, **+11–21% at 32K**, consistently across three
configs. The Vulkan path barely moved, pointing at HIP DMA translation overhead as the mechanism.
Since long-context prefill is this box's whole bottleneck, that's a free win — **confirmed at
128K** (`..\llm-bench\results-strix-halo-rocm.md`, 2026-07-15): Q8_0 `pp131072` went 109→152 t/s
(+40% vs the old performance-mode run), i.e. **128K TTFT ~14.4 min**; decode unchanged. The 128K
TTFT numbers elsewhere in this README predate the change and read conservative.
**Recommendation for every Strix Halo owner: disable IOMMU.** On Windows the only way is BIOS
(IOMMU / AMD-Vi under chipset/advanced settings); on Linux either BIOS or the `amd_iommu=off`
kernel parameter (GRUB).

The genuinely interesting lane for this box is `Q8_0_ROCMFPX_AGENT` — 8-bit (so no quality
give-up) with a preset that claims to protect tool-calling/JSON coherency, which is exactly the
Hermes workload. **Availability check (2026-07-15):** for vanilla Qwen3.6-27B there is NO
published AGENT quant. Q6/Q8 ROCmFPX for the vanilla base do exist
(`philtheriver/Qwen3.6-27B-ROCmFPX`: Q6 24 GB, Q8 29.4 GB; `1337Hero/...Q8_0-ROCMFPX`) — but
**without MTP heads**, which disqualifies them here: by the bandwidth rule they'd decode at
~8.3 / ~6.7 t/s vs our Q8_0+MTP ~17.5 t/s — a 2× regression for at best a marginal format gain.
AGENT variants are only published on the Qwopus coder fine-tune (different base). The only path
to AGENT + MTP + vanilla Qwen is self-quantizing the BF16 source of the unsloth MTP pack:
`bin-rocmfpx\llama-quantize.exe <src-BF16.gguf> <out.gguf> Q8_0_ROCMFPX_AGENT`.

A/B harness: `scripts\rocmfpx-ab.ps1` → `results\rocmfpx-ab.csv`. It runs four configs so the
*format* effect can be separated from the *runtime* effect: Q8_0 on the production build (baseline),
Q8_0 on the fork (control), ROCmFP4 on the fork (ROCm0), and ROCmFP4 on the fork (Vulkan0).

**Pros / cons (all measured on this box unless marked "claimed"):**

| ✅ Advantages                                                        | ❌ Disadvantages                                                       |
|----------------------------------------------------------------------|------------------------------------------------------------------------|
| FP4: 1.8× decode at ≤32K (13.7 vs 7.7 t/s), model half the size      | FP4: −1.7% PPL vs Q8_0; decode edge narrows at 128K (1.79×→1.52×, KV traffic) |
| Runs standard GGUFs too — and +5–7% prefill on Q8_0 vs lemonade build | Its GGUFs are **incompatible with everything else** (LM Studio, stock llama.cpp) |
| Both ROCm0 + Vulkan0 in one build                                     | No releases, no Windows build script — built from source, MSVC 14.44 pin |
| `_AGENT` presets for tool-calling coherency (claimed)                 | AGENT claim unmeasured; not published for our base model (self-quant from BF16 needed) |
| Decode-tune profiles for Strix kernel experiments                     | One-man fork — sustainability/maintenance risk vs upstream llama.cpp    |

**Bottom line — what ROCmFPX is (and is not) good for on this box.** The project's whole value
proposition is the 3–4-bit lane: make Strix-class machines fast by shrinking the weights. This
workload already measured (§7) that smaller weights buy **nothing** at 128K (KV-dominated decode,
quant-independent prefill) and it is quality-first Q8 — so the FP4 lane is not needed here. What
the fork still offers a Q8 user, in order of realism:

1. **A slightly faster runtime for standard GGUFs — but only without MTP.** It reads normal
   quants, and our Q8_0 ran +5–7% faster prefill on it than on the lemonade build (327 vs 305 t/s
   @16K, raw decode identical). **However, with MTP enabled the fork's speculative path is
   clearly slower: 16.3 vs 20.9 t/s TG on the identical Q8_0 model + prompt (temp 0).** Measured
   mechanism (same 2000-token output): the fork drafts far more conservatively — 1381 draft
   tokens vs lemonade's 2119 — with much higher acceptance (93.1% vs 69.3%), yet fewer of the
   output tokens come from drafts (64% vs 73%) and per-step overhead is higher. High acceptance,
   worse throughput: it under-speculates. This more than offsets the prefill edge → **serve Q8
   on the lemonade build (`Serve-Qwen.ps1`); use the fork only for ROCmFPX-format models.**
2. **`Q8_0_ROCMFPX_AGENT`** — 8.25 bpw with allegedly protected tool-calling/JSON tensors. At
   8 bit the headroom over plain Q8_0 is tiny and the claim is unmeasured; for our base model the
   file doesn't exist (would require self-quantizing from a BF16 source). Experiment, not a plan.
3. **The FP4 lane** for speed-first use: −1.7% PPL buys 1.8× decode at short context and a
   measured **1.52× raw decode even at 128K** (9.19 vs 6.06 t/s at depth). For the quality-first
   agent Q8+MTP still wins the trade, but FP4+MTP at 128K is an open speed experiment (see Open
   questions).

**Production config (§5, Q8_0 + MTP on ROCm 7) is unchanged by all of the above.**

## Head-to-head vs LM Studio (same model Q8_0 MTP) — full parity

`scripts\compare-prefill-vs-lmstudio.ps1` + `compare-decode-vs-lmstudio.ps1`. Our ROCm 7
llama-server (:8080) vs LM Studio (:1234), identical `Qwen3.6-27B-Q8_0` MTP, same client-side
method, temp 0:

| Metric                    | Our server | LM Studio | Diff |
|---------------------------|-----------:|----------:|:----:|
| Prompt processing (32K)   |  264.0 t/s | 266.4 t/s | ~1%  |
| Token generation (decode) |  14.56 t/s | 15.09 t/s | ~3%  |

**Full parity** — differences are noise. LM Studio uses a comparably fresh ROCm build (no prefill
advantage either way). The earlier "LM Studio is faster" was purely quant: we were unknowingly on
Q8_K_XL (33 GB) while LM Studio ran Q8_0 (26.6 GB). Switching our default to Q8_0 gave the predicted
~1.27× decode boost (13.83 → 17.55 t/s server-internal = the 33.3/26.6 size ratio). Our advantage is
form factor: a headless API server the Hermes agent hits from another PC, at LM-Studio-equal perf.

## Scripts

Serving / tooling (repo root):

- `Serve-Qwen.ps1` — OpenAI-compatible API server (`-Runtime rocm7 | rocmfpx`); llama-server's
  chat WebUI is on the same port (http://localhost:8081).
- `Serve-Q8-Fork.ps1` — one-shot: the production **Q8_0 + MTP** model on the **fork** runtime
  (WebUI + the fork's small prefill edge). Stops any running llama-server first.
- `Start-InferenceUI.ps1` — lightweight timing playground UI (TTFT / prefill / decode per
  request) on :8082, pointed at the running server.
- `Setup-ROCmFPX.ps1` / `Get-ROCmFPXModel.ps1` — build the ROCmFPX fork / fetch its models (§8).

Benchmarks (`scripts\`):

- `scripts\tune-prefill.ps1` — short-prefill hipBLASLt / micro-batch sweep (done; negative result).
- `scripts\longctx-prefill.ps1` — prefill throughput curve 4K→128K on Qwen 27B (the real TTFT).
- `scripts\rocmfpx-ab.ps1` — ROCmFPX fork + ROCmFP4 format vs the production ROCm 7 build + Q8_0 (§8).
- `scripts\rocmfpx-128k.ps1` — the 128K points: FP4 vs Q8 prefill + true decode-at-depth on the fork (§8).

Results land in `results\`.

## Open questions / next

- Long-context prefill curve (running) → real 128K TTFT number.
- Pull a TheRock gfx1151 nightly llama.cpp build and re-run the long-ctx curve vs b9910.
- Stand up lucebox ROCm and A/B **accuracy + TTFT** on real 100K agent traces (not just NIAH).
- **ROCmFPX (§8):** ~~128K points~~ **done** (`scripts\rocmfpx-128k.ps1`): FP4 decode edge
  narrows to 1.52× but survives. Follow-up worth measuring: **FP4 + MTP at 128K** (llama-cli or
  server-side, llama-bench can't do MTP) — could beat the production Q8+MTP 9.9 t/s, at −1.7% PPL.
- **ROCmFPX (§8):** quantize our own `Q8_0_ROCMFPX_AGENT` from a BF16 Qwen3.6-27B source and A/B it
  against Q8_0 on **agent tool-calling quality**, not just t/s — that preset is the only ROCmFPX
  lane that isn't already ruled out by the bandwidth wall.
- ROCmFPX decode-kernel tuning profiles (`Setup-ROCmFPX.ps1 -Tune rocmfpx-strix-nwarps2` etc.) are
  untested; only `stable` has been built.



