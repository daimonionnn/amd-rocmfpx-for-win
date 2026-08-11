# AMD ROCmFPX for Windows 🏆

Native **Windows** tooling for fast local LLM inference on **AMD Strix Halo** (Ryzen AI MAX,
gfx1151) — and the only Windows build of the [ROCmFPX](https://github.com/charlie12345/ROCmFPX)
llama.cpp fork (AMD-specific GGUF weight formats; upstream ships no releases and Linux-only
build scripts).

Everything is plain PowerShell on Windows 11. Most comparable tooling (vLLM/SGLang serving
stacks, ROCm build scripts, bench harnesses) is Linux-first; this repo fills that gap. Tested
and optimized on a Minisforum MS-S1 MAX (128 GB unified LPDDR5X), usable on other GPUs too
(CUDA / Vulkan / CPU backends included).

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

- **Decode is purely memory-bandwidth-bound** on this APU (`t/s × model-GiB ≈ constant`) —
  bigger quants decode proportionally slower, and no power mode changes that.
- **Prefill is quant-independent** — Q8 prefills as fast as Q4, so quality costs nothing on TTFT.
- **At 128K, Q4 and Q8 even decode at the same speed** (the f16 KV cache dominates memory
  traffic) → run **Q8 + MTP** for quality at no long-context speed cost.
- **ROCmFP4** (the fork's 4-bit format): decode 1.79× ≈ exactly its size ratio, −1.7% perplexity
  vs Q8 — a smaller model, not a faster format; its niche is short-context interactive use.
- **Third-party ROCmFPX "quality" quants don't change this** — the ciru-ai ROCmFP6 STRIX QUALITY
  release (reviewed 2026-08-09) is slower than our config where the numbers are comparable, caps
  at 64K, and its quality edge is inside the noise of its own eval. **Production stays Q8.**

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
