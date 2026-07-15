# AMD Strix Halo ROCm Results

ROCm/HIP benchmark notes for the AMD Strix Halo machine (Minisforum MS-S1 MAX).

> ## ⚠️ Era marker — BIOS 1.06 + IOMMU **ON**
>
> Every result below dated **2026-07-10 … 2026-07-12** was measured on **BIOS 1.06 with IOMMU
> enabled**. On 2026-07-14 the box moved to **BIOS 1.08 + IOMMU disabled**, which lifted
> **prefill/prompt-processing by ~10–20% at longer contexts** (decode is unaffected — measured in
> `..\llm-inference\results\rocmfpx-ab-iommu-on.md` vs `-off.csv`). So the pp numbers below are
> almost certainly **10–20% pessimistic** vs the current machine state; tg numbers still stand.
> In the raw CSVs the eras are separable by the `run_time` column (before/after 2026-07-14).
>
> This also plausibly explains the odd Q8_0 `pp131072` spread in the long-context section
> (137 vs 109 t/s between two runs) — IOMMU-era long-context prefill was exactly this noisy.
>
> **Recommendation for all Strix Halo owners: disable IOMMU in BIOS** (see README).

## Current machine / run state

- Date: **2026-07-10**
- CPU/APU: **AMD RYZEN AI MAX+ 395 w/ Radeon 8060S**
- GPU backend: **llama.cpp b9910 HIP / ROCm** (`ggml-hip.dll`)
- Memory detected by the harness: **128 GB** (`16+16+16+16+16+16+16+16`) @ **8000 MT/s**
- Thread count used: **16**
- Flash Attention: **enabled**
- MTP Max Draft Tokens: **4**

> The baseline numbers below were captured with the machine in a **quiet configuration**.
> Follow-up runs in **performance mode** (2026-07-10 20:21) and **balanced mode**
> (2026-07-12 08:35) were run with the same parameters (`-t 16`, `-p 512`, `-n 128`,
> `-r 3`, FA on). See [Machine-mode comparison](#machine-mode-comparison).

## Results

### 1. Gemma 3 27B QAT Q4_0 (`rocm-gemma3-27b`)

`llama-bench`, `-ngl -1`, `-fa on`, `-p 512`, `-n 128`, `-t 16`, `-r 3`

| Metric  | Result               |
|---------|----------------------|
| `pp512` | **382.49 ± 1.93 t/s** |
| `tg128` | **12.96 ± 0.05 t/s**  |

Source CSV: `results\rocm-gemma3-27b_128GB.csv`

### 2. Qwen3.6 35B-A3B Q4_K_M (`rocm-qwen3.6-35b-a3b`)

`llama-bench`, `-ngl -1`, `-fa on`, `-p 512`, `-n 128`, `-t 16`, `-r 3`

| Metric  | Result                 |
|---------|------------------------|
| `pp512` | **1005.33 ± 9.60 t/s** |
| `tg128` | **64.02 ± 0.72 t/s**   |

Source CSV: `results\rocm-qwen3.6-35b-a3b_128GB.csv`

### 3. Qwen3.6 27B MTP Q4_K_XL (`rocm-qwen3.6-27b-mtp`)

`llama-cli`, `-ngl -1`, `-fa on`, `--spec-type draft-mtp`, `--spec-draft-n-max 4`,
`-p 512`, `-n 128`, `-t 16`, `-r 3`

| Rep   | Prompt t/s | Generation t/s |
|-------|------------|----------------|
| 1     | 44.80      | 23.20          |
| 2     | 44.10      | 23.20          |
| 3     | 45.70      | 23.90          |
| Avg   | **44.87**  | **23.43**      |

Source CSV: `results\rocm-qwen3.6-27b-mtp_128GB.csv`

## Machine-mode comparison

Same exact benchmark set in each machine profile (`-t 16`, `-p 512`, `-n 128`, `-r 3`, FA on):

| Machine mode | Gemma pp512 | Gemma tg128 | Qwen pp512 | Qwen tg128 | MTP prompt avg | MTP gen avg | Notes            |
|--------------|-------------|-------------|------------|------------|----------------|-------------|------------------|
| Quiet        | 382.49      | 12.96       | 1005.33    | 64.02      | 44.87          | 23.43       | baseline run     |
| Balanced     | 411.47      | 13.14       | 1005.96    | 56.83      | 38.83          | 24.97       | 2026-07-12 08:35 |
| Performance  | **418.01**  | **13.17**   | **1107.41**| **63.94**  | **48.57**      | **25.10**   | 2026-07-10 20:21 |

### Balanced vs. Quiet (delta)

| Metric         | Quiet   | Balanced | Δ abs    | Δ %       |
|----------------|---------|----------|----------|-----------|
| Gemma pp512    | 382.49  | 411.47   | +28.98   | **+7.6%** |
| Gemma tg128    | 12.96   | 13.14    | +0.18    | +1.4%     |
| Qwen pp512     | 1005.33 | 1005.96  | +0.63    | +0.1%     |
| Qwen tg128     | 64.02   | 56.83    | -7.19    | **-11.2%**|
| MTP prompt avg | 44.87   | 38.83    | -6.04    | **-13.5%**|
| MTP gen avg    | 23.43   | 24.97    | +1.54    | +6.6%     |

### Balanced vs. Performance (delta)

| Metric         | Performance | Balanced | Δ abs    | Δ %       |
|----------------|-------------|----------|----------|-----------|
| Gemma pp512    | 418.01      | 411.47   | -6.54    | -1.6%     |
| Gemma tg128    | 13.17       | 13.14    | -0.03    | -0.2%     |
| Qwen pp512     | 1107.41     | 1005.96  | -101.45  | **-9.2%** |
| Qwen tg128     | 63.94       | 56.83    | -7.11    | **-11.1%**|
| MTP prompt avg | 48.57       | 38.83    | -9.74    | **-20.1%**|
| MTP gen avg    | 25.10       | 24.97    | -0.13    | -0.5%     |

### Performance vs. Quiet (delta)

| Metric         | Quiet   | Performance | Δ abs    | Δ %      |
|----------------|---------|-------------|----------|----------|
| Gemma pp512    | 382.49  | 418.01      | +35.52   | **+9.3%**  |
| Gemma tg128    | 12.96   | 13.17       | +0.21    | +1.6%    |
| Qwen pp512     | 1005.33 | 1107.41     | +102.08  | **+10.2%** |
| Qwen tg128     | 64.02   | 63.94       | −0.08    | −0.1%    |
| MTP prompt avg | 44.87   | 48.57       | +3.70    | **+8.2%**  |
| MTP gen avg    | 23.43   | 25.10       | +1.67    | **+7.1%**  |

This matches the earlier hypothesis: performance mode helps **prompt/prefill first** (compute-bound,
~+8–10% across all three models). Pure bandwidth-bound generation barely moves — Qwen MoE `tg128` is
flat (−0.1%) and dense Gemma `tg128` gains only +1.6%, because token generation on this APU is limited
by the shared-memory bandwidth, not clocks. The one generation path that does gain is **MTP** (+7.1%):
speculative draft-MTP decoding does extra compute per accepted token, so a higher power ceiling shows up
there where it doesn't in plain autoregressive `tg`.

### Performance-mode source CSVs

Quiet-mode raw rows were preserved before this run:

- `results\rocm-gemma3-27b_128GB_quiet.csv`
- `results\rocm-qwen3.6-35b-a3b_128GB_quiet.csv`
- `results\rocm-qwen3.6-27b-mtp_128GB_quiet.csv`

The live `results\rocm-*_128GB.csv` files now contain quiet (20:00–20:03), performance
(20:21–20:23), and balanced (2026-07-12 08:35–08:37) runs, separable by the `run_time` column.

## Re-run after BIOS 1.08 + IOMMU OFF (2026-07-15)

Same benchmark set (`-t 16`, `-p 512`, `-n 128`, `-r 3`, FA on), clean GPU (no llama-server or
other LLM services running — deliberately, to also rule out the "background services" hypothesis
from the earlier anomaly hunting). Machine mode: **performance** (confirmed by the user).

| Metric         | Old performance (1.06/IOMMU on) | New run 1 (1.08/IOMMU off) | Δ %        |
|----------------|--------------------------------:|---------------------------:|-----------:|
| Gemma pp512    |                          418.01 |                 **435.19** | **+4.1%**  |
| Gemma tg128    |                           13.17 |                      13.33 | +1.2%      |
| Qwen pp512     |                         1107.41 |                    1048.18 | −5.3%      |
| Qwen tg128     |                           63.94 |                      58.13 | **−9.1%**  |
| MTP prompt avg |                           48.57 |                      39.87 | **−17.9%** |
| MTP gen avg    |                           25.10 |                      23.37 | −6.9%      |

Mixed picture: dense Gemma prefill gains as the IOMMU A/B predicts, but the MoE and MTP rows sit
**closer to the old *balanced* row than the old performance row**. A second run reproduced the
first almost exactly (Gemma 437.6 / 13.44; Qwen MoE 1060.7 / 57.83; MTP prompt avg 37.3, gen
23.3), so this is **not noise — the current machine state is stable**, and vs the old
performance row it is: dense prefill **up ~+4–5%**, dense tg flat, **MoE tg128 down ~−9%
(64→58), MTP prompt down ~−20%**.

**Most likely explanation: the BIOS 1.06→1.08 flash changed (or reset) performance-mode power
behavior.** BIOS updates commonly reset/alter power-profile tables, so "performance" before and
after the flash need not be the same power state — worth re-checking the mode setting in
BIOS/vendor tooling after any flash. The IOMMU-off gain itself is real but context-dependent:
small at 512-token prompts (+4–5% here), large at long context (+11–21% at 32K, per the
dedicated A/B) — and long context is what the real workload cares about.

Two of the old open questions can be closed:

- **"Was the odd slow run caused by background services (SQL Server, Defender)?"** — Partially
  moot: the dominant distortion of that era was IOMMU-on long-context prefill noise (±14% at
  32K, −20% swings at 128K). Still, benchmark on an idle box; an *active* LLM server or heavy
  service on the same GPU/memory bus absolutely can skew a run.
- **"Does the 64/96 GB dedicated-VRAM split matter for models smaller than the split?"** — No
  reason it should: llama.cpp HIP on Strix Halo uses unified memory (GTT), so the BIOS carve-out
  only changes what is *reported* as dedicated. The earlier suspect results are explained by the
  IOMMU-era prefill noise; no VRAM-split re-test is needed unless a model exceeds the carve-out.

## Long-context 128K check (Balanced vs Performance)

Run date: **2026-07-12**, using `llama-bench` from `..\llm-bench\bin\` (build `049326a`).

Workload:
- `pp131072` (prefill-only, 128K prompt)
- `tg128@128k` (generation after 128K prefill)
- `tg128-fresh` (generation from empty KV)

### Balanced (run_time: 2026-07-12 16:08:56)

| Model   | pp131072   | tg128 @128K | tg128 fresh |
|---------|-----------:|------------:|------------:|
| Q4_K_XL | 115.45 t/s | 11.85 t/s   | 11.90 t/s   |
| Q8_0    | 137.14 t/s | 7.58 t/s    | 7.55 t/s    |

### Performance (run_time: 2026-07-12 19:12:21)

| Model   | pp131072   | tg128 @128K | tg128 fresh |
|---------|-----------:|------------:|------------:|
| Q4_K_XL | 111.79 t/s | 12.07 t/s   | 12.15 t/s   |
| Q8_0    | 108.95 t/s | 7.58 t/s    | 7.56 t/s    |

### Performance vs Balanced (delta)

| Model   | Metric       | Balanced | Performance | Δ abs   | Δ %      |
|---------|--------------|---------:|------------:|--------:|---------:|
| Q4_K_XL | pp131072     | 115.455  | 111.786     | -3.669  | -3.18%   |
| Q4_K_XL | tg128@128k   | 11.854   | 12.074      | +0.220  | +1.86%   |
| Q4_K_XL | tg128-fresh  | 11.904   | 12.153      | +0.248  | +2.09%   |
| Q8_0    | pp131072     | 137.136  | 108.953     | -28.183 | -20.55%  |
| Q8_0    | tg128@128k   | 7.580    | 7.584       | +0.004  | +0.05%   |
| Q8_0    | tg128-fresh  | 7.552    | 7.556       | +0.004  | +0.06%   |

Note on repetitions: `-r 1` means **one repetition** (single measurement pass). `-r 3` would run the same benchmark three times and report mean/stddev, which is more stable for comparison.

Source CSV: `..\llm-inference\results\longctx-balanced-128k.csv`.
