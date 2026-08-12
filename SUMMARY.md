# What actually moves the needle on Strix Halo

The condensed version of everything measured in this repo. Machine: Minisforum MS-S1 MAX —
Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151), 128 GB LPDDR5X, Windows 11. Workload: a Hermes agent
running 100K+ token contexts on Qwen3.6-27B, quality first.

Every figure below was measured on this one machine. Method, raw CSVs and per-scenario results are
in [llm-inference/README.md](llm-inference/README.md) (§1–§11) with the scripts that produced them
in [llm-inference/scripts/](llm-inference/scripts/). Also published as a
[one-page visual reference](https://claude.ai/code/artifact/8b464808-be9d-4f20-bf50-59926ed92976).

Most tuning ideas that sound promising turned out not to matter. This page is the short version of
which ones did.

---

## Run this

```powershell
.\llm-inference\Serve-Qwen.ps1
```

> **The quant choice changed on 2026-08-12.** This repo assumed "quality-first means Q8" and never
> measured it on anything that sees agent behaviour. Two independently authored tool-calling evals
> now put **Q4_K_M at or above Q8_0** — 41% smaller, slightly faster, far more room for KV cache.
> Details and the caveat below. `Serve-Qwen.ps1` still defaults to Q8_0 pending a long-context
> check; override with `-Model ...Qwen3.6-27B-Q4_K_M.gguf`.

| | | |
|---|---|---|
| **Model** | `Qwen3.6-27B-Q8_0` | 27.04 GiB, unsloth MTP pack |
| **Runtime** | ROCm 7 gfx1151 | lemonade build, `-dev ROCm0` |
| **KV cache** | f16 | Q8 KV buys nothing here |
| **Draft depth** | `--spec-draft-n-max 6` | +2.8% over the old default of 4 |
| **Draft gate** | leave `--spec-draft-p-min` at 0.00 | LM Studio's 0.75 costs 4–15% decode |
| **Context** | 262 144 | at a 64/64 BIOS split; 204 800 on 96/32 |
| **BIOS** | IOMMU **off**, 64 GB / 64 GB split | worth more than every runtime flag combined |

Measured on that config: **128K TTFT ~14.4 min** (was ~68 min before tuning), **decode 13.9 t/s at
128K**, **20.1 t/s at 32K**, 19–20 t/s fresh at 262K.

---

## How this machine behaves

Six measured rules. They explain why the config above looks the way it does — and why most
"improvements" don't work.

**`t/s × GiB ≈ 198` — decode is pure memory bandwidth.** Constant across Q4, Q6 and Q8; about 83%
of the 256 GB/s LPDDR5X ceiling. A bigger quant decodes proportionally slower and no power mode
changes it. The bursty 100%→0% GPU pattern is memory starvation, not throttling. (§4)

**Prefill ignores weight size.** At 128K, Q8 prefills at 152.7 t/s and Q4 at 150.6. Quality costs
nothing on time-to-first-token — the entire penalty for a bigger quant lands in decode. (§3, §8)

**IOMMU off is +40% prefill at 128K.** Free, and it grows with context: +1% at 4K, +6% at 16K,
+11–21% at 32K. Decode unaffected. On Windows there is no way to do this except in BIOS; on Linux,
BIOS or `amd_iommu=off`. (§8)

**Past 64K the KV cache is the model.** The f16 KV cache reaches ~35 GB at 128K, outweighing the
weight difference between quants — so a smaller model stops helping. This is why the 4-bit lane,
compelling on paper, buys nothing at the depth this agent runs. (§7)

**Speculative decoding is the only lever that beats the ceiling.** MTP processes several tokens per
weight stream: 2.11× at 32K, 1.83× at 128K. Everything else works *inside* the bandwidth wall; this
one steps around it. (§5)

**Deep context is gated by host RAM, not VRAM.** The ROCm driver misplaces roughly half the KV cache
to host memory regardless of BIOS split, so host RAM decides how deep you can go. A 64/64 split is
counterintuitively better than 96/32 for 128K+ work, despite the smaller nominal GPU pool.
(Open questions section)

---

## Choosing a quant

Seven quants, same runtime, two independently authored tool-calling evals plus a speed sweep.

| Quant | Size | tool-eval-bench | BFCL | Served decode @16K | Verdict |
|---|---:|---:|---:|---:|---|
| **Q4_K_M** | **15.93 GiB** | **86.3** ✅ | **84.58%** | 20.0 t/s | **Best balance** |
| Q8_0 | 27.05 GiB | 86.3 ✅ | 82.50% | 18.5 t/s | Conservative default |
| Q6_K | 21.31 GiB | 88.1 ✅ | — | 19.8 t/s | Fine, untested on BFCL |
| UD-Q6_K_XL | 24.23 GiB | 88.1 ✅ | — | 17.8 t/s | Fine, no advantage over Q6_K |
| Chadrockv2 FP6 | 23.47 GiB | 85.7 ✅ | — | 19.8 t/s | No speed edge, needs the fork — pointless |
| UD-Q4_K_XL | 16.68 GiB | 85.7 ❌ | — | 20.0 t/s | Fails the safety gate |
| **ROCmFP4** | 15.70 GiB | **82.7** ❌ | **80.83%** | **26.2 t/s** | **Fastest, and last on both evals** |

✅/❌ = the harness's prompt-injection safety gate. Details in §12–§15.

**Bit-width does not order these results.** The 4→8-bit range spans 5.4 points and Q6 sits above
Q8. That says as much about the evals' resolution as about the quants.

> ⚠️ **And most of these gaps cannot be attributed to the format at all.** Three builds of the
> *same* `Q4_K_M` of the *same* model — unsloth, mradermacher, and mradermacher's imatrix
> version — span **83.9 to 87.5**, i.e. 3.6 points. That is exactly as wide as the
> ROCmFP4-to-Q4_K_M gap. **Who built the file moves the score as much as which format it uses.**
>
> So: Q4_K_M ≈ Q8_0 survives (identical scores, and Q8_0 is deterministic where builder barely
> applies). "ROCmFP4 is worse *because of its format*" does not — it is worse *as a file*, by an
> amount a different builder could produce. **Treat gaps under ~4 points as unattributable.**
> (§16)

### Q4_K_M is the surprise

Two unrelated benchmarks put it at or above Q8_0, at 59% of the size and slightly faster, with
11 GiB more room for KV cache — which at 128K+ is this box's actual constraint. It is a standard
GGUF: no fork, runs in LM Studio, runs anywhere.

The caveat that keeps `Serve-Qwen.ps1` on Q8 for now: **both evals ran at 32K context**, and the
production workload is 128K+. Quantization damage plausibly shows up more at depth, and neither
set tests long-context recall.

### ROCmFP4 is fast, and it costs something — but "how much" is unresolved

`Qwen3.6-27B-MTP-ROCmFP4-STRIX-imatrix-embF16-headQ6` genuinely is the fastest model measured
here — §12 found a **real 12–14% kernel edge** over same-size conventional quants, not just a
size effect, plus better draft acceptance. Served, that compounds to **+42% over Q8_0**. That part
is solid: it is a speed result, measured directly, and builder variation does not touch it.

The quality side is weaker than it first looked. It is last on both tool-calling evals and fails
the prompt-injection gate — but by 3.6 points, and §16 showed builder alone produces that spread.
The honest statement is *this file* underperforms, not *this format*. Reasonable for speed-first
short-context work; for an agent with consequences, use something whose quality you have reason
to trust rather than something merely unmeasured.

### The safety gate is not a property of a model

TC-60 (prompt injection hidden in tool output) passes on mradermacher's build and fails on
unsloth's build of the same quant; unsloth passes it with MTP and fails without. Three answers to
one scenario from essentially the same model. The failure sits close enough to a decision boundary
that numerical noise moves it — which is a reason to gate outbound actions outside the model, not
to shop for a build that happens to pass. (§11, §16)

### Never compare quants across runtimes

The first FP4 comparison ran FP4-on-fork against Q8-on-lemonade: 83 vs 83, five scenarios worse and
five better — which reads as "no difference". It is two effects of similar size cancelling out:

```
Q8 lemonade 82.7 ──(+3.6 runtime)──► Q8 fork 86.3 ──(−3.6 quant)──► FP4 fork 82.7
```

Changing the runtime moves the result as much as 4-bit quantization does. A same-config repeat of
the full suite came back **84/84 identical**, so the instrument is deterministic and these deltas
are exact — but only when suite composition is held constant. Subset re-runs are not valid
controls. (§10)

---

## Ideas that did not survive measurement

| Idea | Why it's closed |
|---|---|
| ~~**ROCmFPX 4-bit formats have no kernel edge**~~ | **Wrong — corrected in §12.** That conclusion compared ROCmFP4 against *Q8*, a different size class. Against same-size conventional quants it is genuinely **12–14% faster on raw decode**, because K-quants pay a dequantization cost that grows as bits shrink and ROCmFP4 does not. The lane is closed on *quality* (§13, §14), not on speed. |
| **ROCmFP6 "quality" recipes** | Now measured, not just reviewed: Chadrockv2 FP6 scores fine (85.7, gate passed) but has **zero kernel edge** over Q6_K and a 26% prefill deficit — while requiring the fork. Nothing to gain. (§12, §13) |
| **`Q8_0_ROCMFPX_AGENT`** | The preset promotes nothing above Q8 — it shields 194 of 506 tensors from a cheaper format that plain Q8_0 never uses. Equal or lower precision on every tensor. A `--dry-run` settled it in 176 ms. (§8) |
| **Q8 KV cache** | Zero prefill gain over f16 (prefill is compute-bound) and it crashed at 128K. Only benefit is memory. Revisit only if a fully filled 262K context ever needs the headroom. (§3) |
| **Vulkan backend** | Upstream's "Vulkan decodes better" holds, but by 4% — while costing 36% of prefill at 32K. Prefill is the entire bottleneck. (§8) |
| **Short-prefill tuning** | Forcing hipBLASLt does nothing; micro-batch sweeps buy 3% at best. The build already dispatches to tuned gfx1151 kernels. (§1) |

For reference, our server measured at **full parity with LM Studio** on the identical model —
prefill 264.0 vs 266.4 t/s, decode 14.56 vs 15.09. The advantage is form factor, not speed.

---

## Open, and worth the time

**Should serving move to the ROCmFPX fork?** On identical Q8_0 weights the fork scored 86.3 against
the production build's 82.7 on tool-calling — 5 scenarios better, 1 worse — and passed the
prompt-injection scenario the production build failed. The two are speed-equivalent at serving
depth (128K: 14.2 vs 13.9 t/s decode, 139.5 vs 138.8 prefill). Different kernels round differently,
which is enough to change a sampled token and the whole multi-turn interaction after it.

Not acted on: 84 fixed scenarios measure that set exactly, not tool-calling in general, and the fork
is a one-person project with no releases that must be built from source. Confirm on a second
scenario pack first, and fix `Serve-Qwen.ps1`'s context auto-pick — it keys off runtime rather than
model size, so Q8-on-fork would get 262144 regardless of BIOS split.

**Reasoning off.** 331K tokens across 84 scenarios at 0.42 token efficiency — most generation goes
to reasoning, not answers. Untested, and plausibly a bigger lever on real agent latency than
anything measured so far. The eval harness has `--no-think`.

**Draft depth on agent content.** The n-max sweep ran on prose. Measured acceptance on agent-shaped
content is 0.654 with a mean draft length of 4.92 out of 6 — drive the retest through
`llama-server`, which logs acceptance per request; `llama-cli` does not.

---

## One thing tuning cannot fix

In the `TC-60` scenario a first-turn tool response carries hidden instructions inside ordinary
weather data. Several turns later the model acts on them, adding an attacker's address as BCC to an
outgoing email. Graded critical, and it fails the harness's safety gate.

Reproduced identically on Q8_0 and ROCmFP4, and again on a same-config repeat — a property of
Qwen3.6-27B, not of quantization or settings. The fork runtime happened not to trigger it, but a
failure that floating-point rounding can move is one sitting near a decision boundary, so that is
**not** a mitigation.

**Treat anything arriving through tool output as untrusted input, and gate outbound actions that
carry recipients — email, webhooks — outside the model.** (§11)

---

## Where the detail lives

| | |
|---|---|
| [llm-inference/README.md](llm-inference/README.md) | The full findings, §1–§11, with every table and caveat |
| [llm-inference/scripts/](llm-inference/scripts/) | Benchmark harnesses that produced them |
| [llm-inference/results/](llm-inference/results/) | Raw CSVs, logs, and per-scenario eval JSON |
| [llm-inference/LMStudio-Integration.md](llm-inference/LMStudio-Integration.md) | Running the ROCmFPX build inside LM Studio |
| [llm-bench/](llm-bench/) | `llama-bench` harness and curated Strix Halo results |

Measurements are specific to this machine, driver and BIOS. Treat them as a starting point, not
constants.
