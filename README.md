# AMD ROCmFPX for Windows 🏆

Native **Windows** tooling for fast local LLM inference on **AMD Strix Halo** (Ryzen AI MAX,
gfx1151) — and the only Windows build of the [ROCmFPX](https://github.com/charlie12345/ROCmFPX)
llama.cpp fork (AMD-specific GGUF weight formats; upstream ships no releases and Linux-only
build scripts).

Everything is plain PowerShell on Windows 11. Most comparable tooling (vLLM/SGLang serving
stacks, ROCm build scripts, bench harnesses) is Linux-first; this repo fills that gap. Tested
and optimized on a Minisforum MS-S1 MAX (128 GB unified LPDDR5X), usable on other GPUs too
(CUDA / Vulkan / CPU backends included).

**→ [SUMMARY.md](SUMMARY.md) — start here.** Everything measured, condensed: the production config,
how this machine behaves, which quant to pick, and the six tuning ideas that didn't survive
measurement. The detail below and in `llm-inference/` backs it up.

## What's inside

| Folder | What it is |
|--------|------------|
| **[`llm-inference/`](llm-inference/)** — [README](llm-inference/README.md) | **The main project.** ROCmFPX Windows build (`Setup-ROCmFPX.ps1`), headless OpenAI-compatible llama-server with chat WebUI (`Serve-Qwen.ps1`), model fetcher, timing UI — plus all measured findings: the long-context prefill wall, ROCm 7 build gains, MTP speculative decoding, quant trade-offs, ROCmFP4 speed & quality, IOMMU effect. |
| **[`llm-bench/`](llm-bench/)** — [README](llm-bench/README.md) | Companion benchmark harness (llama.cpp `llama-bench`) and provisioning for the shared binaries. Curated Strix Halo results in [results-strix-halo-rocm.md](llm-bench/results-strix-halo-rocm.md). |
| **[LMStudio-Integration.md](llm-inference/LMStudio-Integration.md)** | Run the ROCmFPX build **inside LM Studio** by swapping a backend folder — so ROCmFPX-format GGUFs load in the GUI. Four branches with their failure modes, and the HIP-runtime step most guides omit. |

## Headline results (Qwen3.6-27B, Radeon 8060S, 128 GB)

The journey from stock to tuned, at the workload that hurts — a **128K-token context**:

| Step | 128K prefill | 128K TTFT | Decode @128K |
|------|-------------:|----------:|-------------:|
| Stock llama.cpp HIP build (b9910), Q4 | 32.3 t/s | **~68 min** | ~5–6 t/s |
| + ROCm 7 gfx1151-specific build ([lemonade-sdk](https://github.com/lemonade-sdk/llamacpp-rocm)) | 150.6 t/s | ~14.5 min | — |
| + BIOS: **IOMMU off** (see below) | **152 t/s (Q8!)** | **~14.4 min** | — |
| + MTP speculative decoding (`--spec-type draft-mtp`) | — | — | **~9.9 t/s** (2× plain) |

Other load-bearing findings (measured, details in [llm-inference/README.md](llm-inference/README.md)):

- **Decode is memory-bandwidth-bound** on this APU (`t/s × model-GiB ≈ constant`) — bigger quants
  decode proportionally slower, and no power mode changes that. (Nearly constant: the product
  tilts ~8% from Q8 to Q4 because K-quants pay a dequantization cost. That tilt turned out to
  matter — see below.)
- **Prefill is quant-independent at depth** — at 128K Q8 prefills as fast as Q4, so quality costs
  nothing on TTFT. At 16–32K the spread is real (Q8 is fastest), so this holds where it matters
  and not everywhere.
- **At 128K, Q4 and Q8 even decode at the same speed** (the f16 KV cache dominates memory
  traffic) → the quant choice at long context is about quality and memory headroom, not speed.
- **Bit-width does not predict tool-calling quality.** Across seven quants and two independently
  authored agent evals, the 4→8-bit range spans ~5 points and the ranking ignores precision —
  **Q4_K_M matches Q8_0**. Perplexity said otherwise and perplexity was wrong.

## ROCmFPX — what it actually buys you

This is the repo's headline feature, so here is everything measured about it rather than a
one-liner. ROCmFPX is a llama.cpp fork adding AMD-specific GGUF weight formats — `Q4_0_ROCMFP4`,
`Q6_0_ROCMFPX`, `Q8_0_ROCMFPX` and `_AGENT` presets. Stock llama.cpp cannot read them.

**These are two different formats and they behave nothing alike.** Measured against
*same-size conventional quants* on the same runtime — the comparison that actually tests the
format, which most published numbers do not make:

| | `Q4_0_ROCMFP4` | `Q6_0_ROCMFPX` (Chadrockv2 FP6) |
|---|---|---|
| **Raw decode vs same-size peers** | **+12–14%** — a real kernel edge | **0%** — identical to Q6_K |
| **MTP draft acceptance** | 2.01× vs 1.77–1.84× for peers | 2.42× vs 2.17–2.18× |
| **Served decode @16K** | **26.2 t/s — fastest measured here, +42% over Q8_0** | 19.8 t/s, same as Q6_K |
| **Prefill** | on par with its league | **−26% against Q6_K** |
| **Tool-calling (two evals)** | **last of seven; fails the prompt-injection gate** | fine (85.7, gate passed) |
| **Verdict** | fast, and it costs measurable agent quality | no advantage in either direction |

**Why the kernel edge is real and not just "smaller model".** `t/s × GiB` is not constant across
bit-widths: K-quants pay a dequantization cost that grows as bits shrink — Q8_0 sits at ~208,
Q6_K ~202, Q4_K_M ~191. ROCmFP4 sits at ~217, i.e. **on the Q8 line**, which is exactly what 4-bit
K-quants fail to do. That is genuine kernel work, and it is the strongest technical argument for
the project. It is also invisible if you benchmark against Q8 instead of a 4-bit peer, which is
why this repo missed it for weeks.

**Where it is weak — and how much of that is the format's fault:**

| ❌ | Detail |
|---|---|
| Quality on agent tasks | ROCmFP4 scored last on both tool-call evals and failed the prompt-injection scenario Q8 passes on the same runtime |
| Portability | Its GGUFs load nowhere else — stock llama.cpp, LM Studio and Ollama all refuse them (LM Studio can be made to work by [swapping a backend folder](llm-inference/LMStudio-Integration.md)) |
| Distribution | No releases, no Windows build script upstream — build from source, with an MSVC 14.44 pin |
| Supply | unsloth, mradermacher and bartowski do not make these files and never will. Every ROCmFPX quant in existence comes from a handful of individuals, with per-author recipes that are usually unstated and no independent verification |
| Sustainability | A one-person fork |

**An important caveat on the quality verdict — now measured, not just suspected.** Our ROCmFP4
comes from one author with unknown calibration; the quants it lost to are unsloth's. To find out
how much of that gap is *format* and how much is *builder*, we ran three builds of the same
`Q4_K_M` of the same model:

| unsloth | mradermacher | mradermacher `i1` (imatrix) |
|---:|---:|---:|
| 85.1 | **87.5** | **83.9** |

**3.6 points of spread from builder alone — exactly the size of the gap that made ROCmFP4 look
worse.** (And imatrix calibration made it *worse* here, not better, which is its own surprise.)
So "ROCmFP4 scores lower" is a statement about **this file**, not the format. A better-built
ROCmFP4 could plausibly close it while keeping the kernel advantage; nothing here rules that out.

The speed result is unaffected — 12–14% is a direct measurement of the same file against peers,
and builder variation does not move throughput.

**Fair framing: this is a young format.** The 4-bit kernel work is real, measurable and, on this
hardware, the fastest thing we have run. What it lacks is the ecosystem — reproducible recipes,
multiple independent builders, and the years of quiet bug-finding that make K-quants boring and
trustworthy. Those are solvable with time and attention, not physics. If better ROCmFP4 quants
appear, the speed argument is already made and only the quality question would be left open.

**Today's practical read:** use it for speed-first, short-context, low-stakes work where 26 t/s
beats 18 t/s and a wrong tool call costs nothing. Do not put it behind an agent that can send
email. And re-check when new quants land — the conclusion here is about the files that exist now.

## ⚠️ If you own a Strix Halo machine: disable IOMMU in BIOS

Measured on this box: with IOMMU enabled, **prefill loses ~10–20% at 32K and ~30–40% at 128K
context** (decode unaffected). It's free performance:

- **Windows:** BIOS only — IOMMU / AMD-Vi under chipset or advanced settings → *Disabled*.
- **Linux:** BIOS, or the `amd_iommu=off` kernel parameter (GRUB).

Full A/B: [rocmfpx-ab-iommu-on.md](llm-inference/results/rocmfpx-ab-iommu-on.md) and the
[128K re-run](llm-bench/results-strix-halo-rocm.md).

## Quick start

```powershell
# 1. llama.cpp binaries (ROCm 7 gfx1151 build by default; cuda13/cuda12/vulkan/cpu also available)
.\llm-bench\Setup.ps1

# 2. Serve Qwen3.6-27B Q8_0 + MTP as an OpenAI-compatible API (+ chat WebUI on the same port)
.\llm-inference\Serve-Qwen.ps1          # -> http://localhost:8081/v1

# Optional: build the ROCmFPX fork runtime and try its ROCmFP4 formats
.\llm-inference\Setup-ROCmFPX.ps1       # needs HIP SDK, Vulkan SDK, MSVC, cmake, ninja
.\llm-inference\Get-ROCmFPXModel.ps1
.\llm-inference\Serve-Qwen.ps1 -Runtime rocmfpx
```

## Hardware reference

Minisforum MS-S1 MAX — AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151), 128 GB LPDDR5X
@ 8000 MT/s (unified, ~256 GB/s), Windows 11 Pro, BIOS 1.08, IOMMU disabled.
