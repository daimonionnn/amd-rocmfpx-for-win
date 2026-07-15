# LLM Benchmark Harness — AMD Strix Halo

Companion **benchmark harness** (built on llama.cpp's `llama-bench`) for the main project,
**`..\llm-inference`** — *ROCmFPX for Windows + headless LLM serving*. That repo holds the
ROCmFPX Windows runtime, the llama-server scripts, and all optimization findings; this one
provides the supporting benchmarks and provisions the shared llama.cpp binaries (`bin\`).

**Windows-first.** Everything here is native PowerShell, built and tested on Windows 11 —
primarily for **AMD Strix Halo** (Ryzen AI MAX / gfx1151), but the multi-backend setup
(CUDA/Vulkan/CPU) keeps it usable on other GPUs. Most comparable tooling is Linux-first /
bash-only; this pair of repos is the PowerShell-native alternative.

## Hardware

- **Machine:** Minisforum MS-S1 MAX (AMD Strix Halo), BIOS 1.08, IOMMU disabled
- **APU:** AMD Ryzen AI MAX+ 395 / Radeon 8060S (**gfx1151**), 32 CPU threads
- **RAM:** 128 GB LPDDR5X @ 8000 MT/s (soldered, unified memory, ~256 GB/s ceiling)
- **OS:** Windows 11 Pro

Key measured facts about this box (details in `results-strix-halo-rocm.md` and
`..\llm-inference\README.md`): token generation is **memory-bandwidth bound** (~83% of the
256 GB/s ceiling); long-context prefill is the real bottleneck (O(S²)); the ROCm 7
gfx1151-specific build is up to **4.66× faster** at 128K prefill than the generic HIP build;
MTP speculative decoding ~doubles decode.

## Tests

### ROCm / APU pack (Gemma + Qwen + Qwen MTP) — `Run-RocmTest.ps1`

Runs three tests in sequence:

- **Gemma 3 27B** with `llama-bench`
- **Qwen3.6 35B-A3B** (MoE) with `llama-bench`
- **Qwen3.6 27B MTP** with `llama-cli` speculative mode `draft-mtp`

**Flash Attention** is enabled by default (`-fa on`); the MTP run uses
**Max Draft Tokens = 4** (`--spec-draft-n-max 4`).

Default MTP model path is the LM Studio location:
`C:\Users\matt\.lmstudio\models\unsloth\Qwen3.6-27B-MTP-GGUF`

```powershell
.\Run-RocmTest.ps1
.\Run-RocmTest.ps1 -Device 0 -GpuLayers -1 -MtpMaxDraftTokens 4
```

Outputs (auto-named by detected RAM total):
- `results\rocm-gemma3-27b_128GB.csv`
- `results\rocm-qwen3.6-35b-a3b_128GB.csv`
- `results\rocm-qwen3.6-27b-mtp_128GB.csv`

Long-context benchmarks (prefill curves, decode quant compare, ROCmFPX A/B, perplexity) live
in `..\llm-inference\scripts\`.

## Setup

```powershell
.\Setup.ps1                      # DEFAULT: ROCm 7 gfx1151 build (lemonade-sdk) -> bin\
.\Setup.ps1 -Backend hip -Force  # official ggml-org multi-arch HIP build (older ROCm; control)
.\Setup.ps1 -Backend cuda13      # official CUDA builds - for NVIDIA machines
.\Setup.ps1 -Backend cuda12
.\Setup.ps1 -Backend vulkan
.\Setup.ps1 -Backend cpu
```

The ROCm 7 gfx1151 build is the production choice on this box — measured up to ~4.7× faster
long-context prefill than the generic HIP build (see `..\llm-inference\README.md` §3). The
CUDA/Vulkan/CPU backends are kept so the harness stays usable on other hardware (e.g. NVIDIA).

### ROCmFPX runtime

The ROCmFPX fork build, its models, and the serving scripts live in **`..\llm-inference\`**
(`Setup-ROCmFPX.ps1`, `Get-ROCmFPXModel.ps1`, `Serve-Qwen.ps1`) — they are inference/serving
infrastructure, not benchmarks. See `..\llm-inference\README.md` §8.

## What each column means (llama-bench output)

| column  | meaning                                                                  |
|---------|--------------------------------------------------------------------------|
| `pp512` | prompt processing throughput (compute-bound) at 512-token prompt         |
| `tg128` | **token generation** throughput (memory-bandwidth-bound) for 128 tokens  |
| `t/s`   | tokens per second (± is std-dev across reps)                             |

## Files

```
Setup.ps1                   provision bin\ (ROCm 7 gfx1151 default; hip/vulkan/cpu variants)
Common.ps1                  shared paths, RAM/CPU detection, single-run bench + result logging
Run-RocmTest.ps1            ROCm/APU pack (Gemma + Qwen MoE + Qwen MTP)
results-strix-halo-rocm.md  curated Strix Halo results
bin\                        llama.cpp binaries + DLLs (ROCm 7 gfx1151 build)
bin-official-b9910\         official b9910 HIP build (kept as the "before" control)
results\                    CSV logs, one file per test
```
