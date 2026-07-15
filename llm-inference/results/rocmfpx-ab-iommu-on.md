# ROCmFPX A/B — baseline with **IOMMU ON** (pre-BIOS-change)

Kept as the "before" side of the IOMMU experiment. The "after" side
(`rocmfpx-ab-iommu-off.csv`) differs from this baseline in TWO ways at once, so any delta is
their combined effect, not attributable to IOMMU alone:
  1. BIOS IOMMU: enabled -> **disabled** (expected ~+15%)
  2. BIOS version: **1.06 -> 1.08** (Minisforum MS-S1 MAX, AMD Strix Halo)

Machine state: Radeon 8060S / gfx1151, ROCmFPX fork build `6bf20cd`, `-fa on -ngl -1 -ub 1024`,
f16 KV, no MTP (llama-bench has no speculative support). GPU verified idle (110296/110456 MiB free).

## Numbers

| config       | model                    | device                     | pp4096  | pp16384 | pp32768 | tg128 | tg128 @32K |
|--------------|--------------------------|----------------------------|--------:|--------:|--------:|------:|-----------:|
| `q8-rocm4`   | Q8_0 (27.04 GiB)         | ROCm0, production `bin\`   |   360.4 |   300.8 |   244.6 |  7.57 |       7.14 |
| `q8-fpx`     | Q8_0 (27.04 GiB)         | ROCm0, fork `bin-rocmfpx\` |   365.3 |   308.1 |   252.3 |  7.69 |       7.22 |
| `fp4-fpx`    | Q4_0_ROCMFP4 (15.69 GiB) | ROCm0, fork                | 349.4\* | 306.2\* | 234.2\* | 13.76 |      12.21 |
| `fp4-fpx-vk` | Q4_0_ROCMFP4 (15.69 GiB) | Vulkan0, fork              |   294.3 |   251.9 |   166.9 | 14.31 |      12.55 |

\* The `fp4-fpx` prefill row is from a separate 3-rep re-measure, not from the sweep. The sweep's
2-rep run produced 270 / 231 / **289** — a physically impossible curve (pp32768 > pp16384). See
"Caveats".

## Caveats — read before trusting any single prefill number

1. **`pp32768` has ~±14% run-to-run spread on this box** (measured `234.23 ± 33.79` over 3 reps).
   Thermal/clock behaviour, not a model property. Two reps are not enough; the sweep now uses 3.
2. The sweep that produced this table ran the buggy version of `rocmfpx-ab.ps1`, which invoked
   `llama-bench` **twice** per pass (once `-o md` to print, once `-o csv` to log) and therefore
   logged *different numbers than it printed* — the same 32K prefill measured 200 t/s in one run
   and 289 t/s in the other. Fixed: one run per pass, console rendered from the logged CSV.
3. Decode numbers are stable (spread < 1%) and are the trustworthy part of this table.

## What it says (as of IOMMU ON)

- **The fork costs nothing on standard quants.** `q8-fpx` ≈ `q8-rocm4` within noise (+2-3%), so
  moving to the ROCmFPX runner is free.
- **Prefill is quant-independent**, confirming §3 of the README: ROCmFP4 (349/306/234) ≈ Q8_0
  (365/308/252) within the ±14% noise, despite being 42% smaller.
- **ROCmFP4's whole win is decode, and it is just the bandwidth rule.** 13.76 vs 7.69 t/s = 1.79×,
  and the weight-size ratio is 27.04/15.69 = 1.72×. `t/s × GiB` = 216 (fp4) vs 208 (Q8_0) — only
  ~4% above the memory-bandwidth line. It is a smaller model, not a better one.
- **Vulkan loses.** +4% decode, but −29% prefill at 32K (166.9 vs 234.2). Prefill is the
  bottleneck here → stay on `-dev ROCm0`.
- Per README §7, ROCmFP4's decode edge should **vanish at 128K** (f16 KV cache dominates and is
  quant-independent). The `-Full` run has not been done yet.

## Outcome (after the change — see `rocmfpx-ab-iommu-off.csv`)

Decode: completely flat across all four configs (bandwidth unaffected). Prefill on the ROCm path:
+1% @4K, +6% @16K, +11–21% @32K, consistent across configs; Vulkan path barely moved. Verdict:
IOMMU-off/BIOS-1.08 removes a per-translation DMA overhead that grows with context length — a
free long-context-prefill win, no downside observed. Full analysis in README §8.
