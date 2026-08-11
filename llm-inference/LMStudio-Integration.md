# Running the ROCmFPX build inside LM Studio (Windows)

LM Studio cannot load ROCmFPX-format GGUFs — the custom tensor types (enum IDs 100–117) do not
exist in the upstream llama.cpp it ships. You can, however, replace the binaries inside one of its
backend folders with the build produced by [`Setup-ROCmFPX.ps1`](Setup-ROCmFPX.ps1), which gives
you a native GUI workflow for the same models `Serve-Qwen.ps1 -Runtime rocmfpx` serves headlessly.

This started as a community write-up:
[How to Enable ROCmFPX Acceleration in LM Studio on Windows](https://www.reddit.com/r/StrixHalo/comments/1veavil/how_to_enable_rocmfpx_acceleration_in_lm_studio/)
(r/StrixHalo), which credits this repo for the Windows build. The steps below are that idea plus
what actually breaks — the original omits the HIP runtime dependency, which is the most common
failure, and recommends the backend folder that makes it hardest to fix.

> **Set expectations first.** This is not an "acceleration" patch. §8 of the
> [main README](README.md) measured ROCmFPX's decode advantage as almost exactly its weight-size
> ratio (`t/s × GiB` ≈ 216 vs the 196–208 bandwidth line), i.e. a few percent of kernel edge at
> most — ROCmFP4 is a *smaller model*, not a faster format, and prefill is quant-independent. The
> reason to do this is to **load ROCmFPX-format models in LM Studio at all**. A possible second
> reason is in "Does it change quality?" at the end.

## How LM Studio's backends are put together

Worth understanding before overwriting anything, because a backend folder is not a plain
llama.cpp build. Measured on LM Studio with the ROCm backend `2.28.1`:

```
%USERPROFILE%\.lmstudio\extensions\backends\
├── llama.cpp-win-x86_64-amd-rocm-avx2-2.28.1\   418–423 MB per version
│   ├── ggml-hip.dll              396.79 MB   ← device code for 7 gfx targets
│   ├── llama-server.exe            0.02 MB   ← 20 KB stub, not the real server
│   ├── llama-server-impl.dll       4.77 MB   ← the actual implementation
│   ├── llm_engine.dll, *.node                ← LM Studio's own bindings
│   └── backend-manifest.json
├── llama.cpp-win-x86_64-vulkan-avx2-2.28.1\     55.7 MB
└── vendor\
    ├── win-llama-rocm-vendor-v6\   361.4 MB  ← the HIP runtime lives HERE
    │   └── bin\  amd_comgr_3.dll 109.9, rocblas.dll 39.2, amdhip64_7.dll 17.3,
    │             libhipblaslt.dll 5.7, + Tensile kernels incl. 96 gfx1151 files
    └── win-llama-vulkan-vendor-v2\   1.2 MB  ← no HIP runtime at all
```

Two facts drive everything below.

**LM Studio spawns `llama-server.exe` as a subprocess.** `backend-manifest.json` declares it:

```json
"engine_protocol_server": { "runtime_kind": "llama-server",
                            "executable_relative_path": "llama-server.exe" }
```

Confirmed from a live process — LM Studio launches it with `--model … --port … --ctx-size …
--flash-attn on --spec-type draft-mtp …`. So dropping our real `llama-server.exe` (9.7 MB) in
place of their 20 KB stub works: LM Studio talks to it over HTTP and does not care what produced
the binary. It also means `llm_engine.dll` and the `.node` files are irrelevant to this path —
leave them alone.

**Notably, LM Studio passes no `-dev` flag.** Our [`Serve-Qwen.ps1`](Serve-Qwen.ps1) always forces
`-dev ROCm0`, because §8 measured Vulkan costing −36% prefill at 32K on this box. Our fork build
registers *both* ROCm and Vulkan devices, so without `-dev` the device choice is left to
llama.cpp's enumeration order. Check the loaded-model logs and confirm you got ROCm.

**Our build carries no HIP runtime.** `bin-rocmfpx\ggml-hip.dll` is 78.9 MB against LM Studio's
396.79 MB, because `Setup-ROCmFPX.ps1` stages binaries only — `Serve-Qwen.ps1` prepends
`%HIP_PATH%\bin` to `PATH` at launch (see its `$env:PATH = …` line). **LM Studio does not do
that.** On this machine the HIP SDK is on neither the machine nor the user `PATH`:

```powershell
([Environment]::GetEnvironmentVariable('PATH','Machine') -split ';') -match 'ROCm.*bin'   # → nothing
([Environment]::GetEnvironmentVariable('PATH','User')    -split ';') -match 'ROCm.*bin'   # → nothing
```

So a straight copy of our binaries produces a backend that fails to load `rocblas.dll` /
`amdhip64_7.dll`. This is the step the Reddit guide is missing, and it is why option B below is
the recommended one.

## Which backend folder to overwrite

| Branch | HIP runtime available? | Notes |
|---|---|---|
| **A. ROCm backend** (`…amd-rocm-avx2-*`) | ✅ vendor package is ROCm | Recommended target. Manifest already declares `gfx1151`. |
| **B. ROCm backend + HIP DLLs copied in** | ✅ guaranteed | **Most reliable.** Immune to how LM Studio sets the search path. |
| **C. Vulkan backend** (`…vulkan-avx2-*`) | ❌ vendor is 1.2 MB, no HIP | What the Reddit post suggests. Our ROCm path will not load unless you also solve the runtime. |
| **D. Any backend + `%HIP_PATH%\bin` on system PATH** | ✅ | One-line fix, but changes global environment and needs the HIP SDK installed. |

Prefer **B**. Windows searches the executable's own directory first, so putting the HIP DLLs
beside `llama-server.exe` works no matter what LM Studio does with `PATH`, and it does not require
the HIP SDK to be installed at all — LM Studio already ships every file needed.

Always overwrite an **older** backend version you do not otherwise use (e.g. `2.25.2` while
`2.28.1` stays stock). That keeps a working engine to fall back to and makes "revert" mean
"delete the folder and let LM Studio re-download it".

## Step by step (option B, recommended)

### 1. Build

```powershell
git clone https://github.com/daimonionnn/amd-rocmfpx-for-win
cd amd-rocmfpx-for-win
.\llm-inference\Setup-ROCmFPX.ps1
```

Needs the AMD HIP SDK, Vulkan SDK, MSVC (VS2022) and cmake/ninja. Requirements and the MSVC 14.44
pin are documented inside `Setup-ROCmFPX.ps1`. Output lands in `llm-inference\bin-rocmfpx\`
(14 files, 190 MB).

### 2. Close LM Studio completely

Not just the window — check that no `llama-server.exe` or LM Studio process survives, or the
copy will fail on locked files:

```powershell
Get-Process llama-server, "LM Studio" -ErrorAction SilentlyContinue
```

### 3. Pick a target and back it up — **outside `backends\`**

> ⚠️ **The single most likely way to break this.** LM Studio treats *every subfolder* of
> `extensions\backends\` as an installed backend, including one you created as a backup. A folder
> named `…-2.25.2.backup` registers as a backend that sorts after `…-2.25.2`, and LM Studio will
> happily launch **it** — the untouched stock binaries — which then fail on ROCmFPX tensor types
> with `exitCode=1`. The symptom is maddening because the folder you edited looks perfect:
>
> ```
> 🥲 Failed to load the model
> Engine protocol runtime llama-server for … exited before becoming healthy. exitCode=1, signal=null
> ```
>
> Diagnosed the hard way on this machine: a process capture during the failed load showed
> LM Studio running `llama.cpp-win-x86_64-amd-rocm-avx2-2.25.2.backup\llama-server.exe`.
> Keep backups anywhere except `backends\`.

```powershell
$b = "$env:USERPROFILE\.lmstudio\extensions\backends"
Get-ChildItem $b -Directory | Where-Object Name -match 'amd-rocm'   # choose an older version
$target = "$b\llama.cpp-win-x86_64-amd-rocm-avx2-2.25.2"

# Back up OUTSIDE the backends directory, not next to it
$backups = "$env:USERPROFILE\lmstudio-backend-backups"
New-Item -ItemType Directory -Force -Path $backups | Out-Null
Copy-Item $target "$backups\$(Split-Path $target -Leaf).backup" -Recurse
```

Reverting later means copying that folder back, or simply deleting the modified one and letting
LM Studio re-download the runtime.

### 4. Copy in the ROCmFPX binaries

```powershell
Copy-Item "<repo>\llm-inference\bin-rocmfpx\*" $target -Force
```

This overwrites `llama-server.exe`, `llama.dll`, `llama-common.dll`, `ggml-*.dll` and `mtmd.dll`,
and adds `llama-cli.exe` / `llama-quantize.exe` / `llama-bench.exe`, which do no harm. LM Studio's
`backend-manifest.json`, `display-data.json`, `llm_engine*.dll` and `.node` files stay untouched —
the manifest must keep describing a `llama-server` runtime, which it still does.

### 5. Copy in the HIP runtime (the step the Reddit guide omits)

From LM Studio's own vendor package — no HIP SDK needed:

```powershell
$rv = "$b\vendor\win-llama-rocm-vendor-v6"
Get-ChildItem $rv -File -Recurse |
  Where-Object { $_.Name -match 'gfx1151' -or $_.Name -notmatch 'gfx\d' } |
  Copy-Item -Destination $target -Force
```

That filter takes the architecture-agnostic files plus only the gfx1151 kernels: **156 files,
192 MB**, versus 361 MB for every architecture. `amd_comgr_3.dll` alone is 109.9 MB and is
required. If you would rather use the HIP SDK, the same files live under `%HIP_PATH%\bin`.

> Keeping Tensile kernels for other gfx targets is harmless but pointless on Strix Halo. If you
> want the whole thing, drop the `Where-Object` filter.

### 6. Point LM Studio at it

1. Start LM Studio → **App Settings → Runtime**.
2. Select the version you modified.
3. **Turn Auto-Update off** — otherwise LM Studio will restore its own binaries on the next
   update and the failure will look mysterious.
4. Load a ROCmFPX model (`Get-ROCmFPXModel.ps1` fetches one into `llm-inference\models\`).

### 7. Verify you got what you think you got

Check the loaded-model log for the ROCm device rather than Vulkan, and — since LM Studio passes no
`-dev` — that it did not split across both. A quick sanity check outside LM Studio:

```powershell
$env:PATH = "$($env:HIP_PATH.TrimEnd('\'))\bin;$env:PATH"
.\llm-inference\bin-rocmfpx\llama-quantize.exe --help | Select-String ROCMFP
```

If that lists enums `100`–`117`, the build itself is fine and any remaining problem is in the
LM Studio wiring.

## Variants

**Option A — ROCm backend, no DLL copy.** Try it first if you prefer a smaller diff: do steps 1–4
and skip 5. It depends on LM Studio making its vendor directory visible to the spawned process. If
the model fails to load with a DLL error, do step 5. Nothing is lost by trying.

**Option C — Vulkan backend.** What the original post recommends. Our build does contain
`ggml-vulkan.dll`, so a *Vulkan-only* path can work, but on this hardware that is the slower
device (−36% prefill at 32K, §8), and `win-llama-vulkan-vendor-v2` is 1.2 MB with no HIP runtime —
so the ROCm path inside our build will not load unless you also do step 5 or option D. If you take
this branch, do step 5 as well; there is no advantage over the ROCm folder.

**Option D — HIP SDK on the system PATH.** Add `%HIP_PATH%\bin` to the machine `PATH` and skip
step 5 entirely. One line, but it is global state, it needs the HIP SDK installed, and it can
shadow other ROCm installations. Fine on a dedicated box, worse as published advice.

## Troubleshooting

**`exited before becoming healthy. exitCode=1`** — almost always means LM Studio launched a
*different* binary than the one you edited. Confirm which, by capturing the process during a load:

```powershell
$job = Start-Job { $seen=@{}; $end=(Get-Date).AddSeconds(120)
  while ((Get-Date) -lt $end) {
    Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" -EA SilentlyContinue |
      ForEach-Object { if (-not $seen[$_.ProcessId]) { $seen[$_.ProcessId]=$true
        [pscustomobject]@{ PID=$_.ProcessId; Path=$_.ExecutablePath } } }
    Start-Sleep -Milliseconds 100 } }
& "$env:USERPROFILE\.lmstudio\bin\lms.exe" load <model-key> -c 4096 -y
Stop-Job $job; Receive-Job $job | Format-List
```

`ExecutablePath` tells you the truth in one line. In our case it pointed at the `.backup` folder
(see step 3). Use `lms load` rather than the GUI while debugging — same engine path, and the error
arrives in the terminal.

If it *is* the folder you edited, the next suspects are a missing HIP runtime (step 5) or a model
belonging to a different ROCmFPX fork than the one you built.

**Which model formats your build can read** is broader than the model cards suggest. The ciru
`Chadrockv2 ROCmFP6 STRIX QUALITY` card says it needs their `ciru-ai/ROCmFPX @ rocmfp6-strix-quality`
branch — but it loads fine on a build from `charlie12345/ROCmFPX @ main`. The `118` in that card is
the *quantization recipe* enum used by `llama-quantize`, not a tensor type; the file's tensors are
`Q6_0_ROCMFPX` and `Q8_0_ROCMFPX` (enums 110/111), which the main branch understands. You need
their branch to *create* such a file, not to read one. Check what your build supports with:

```powershell
.\bin-rocmfpx\llama-quantize.exe --help | Select-String ROCMFP
```

## Reverting

Delete the modified folder and let LM Studio re-download that runtime, or copy back the backup you
parked outside `backends\` in step 3. Because you overwrote an older version and left the newest
one stock, LM Studio still has a working engine either way.

## Does it change quality? Measured: yes

§10 of the [main README](README.md) ran an 84-scenario tool-calling benchmark against the *same*
`Qwen3.6-27B-Q8_0` weights on both runtimes:

| | lemonade ROCm 7 | ROCmFPX fork |
|---|---:|---:|
| Score (common 168-point basis) | 82.7 | **86.3** |
| Scenarios better / worse | — | **5 better, 1 worse** |
| TC-60 prompt-injection scenario | ❌ fails | ✅ passes |

A same-config repeat of the suite came back **84/84 identical**, so this is an exact measurement,
not run-to-run variance. Different kernels produce different floating-point rounding, which is
enough to change a sampled token and therefore how a multi-turn tool interaction unfolds.

Two things this does **not** mean. It is not evidence that the fork is *safer* — a
prompt-injection failure that numerical noise can move is a failure sitting near a decision
boundary, and §11 treats the underlying weakness as a property of the model on both runtimes. And
84 fixed scenarios measure this scenario set exactly, not tool-calling in general; the direction
looks solid, the magnitude may not carry to a different pack.

So: a real, measured reason to prefer this build beyond format support — but not a reason to skip
your own testing on your own workload.

## Known gaps

- **Verified working on this machine (2026-08-11):** option B, ROCm backend `2.25.2`, ROCmFP4
  model loaded through `lms load` in 26.8 s at 17.41 GiB. What is *not* verified is sustained
  generation quality or speed through LM Studio — only that it loads and serves.
- LM Studio's MTP toggle ("MTP Speculative Decoding" in advanced load settings) passes
  `--spec-type draft-mtp`; whether its defaults match §9's measured best (`--spec-draft-n-max 6`,
  `p-min` at llama.cpp's 0.00) is unchecked. LM Studio was observed using `n-max 4` with
  `--spec-draft-p-min 0.75`, which §9 measured as 4–15% slower on decode.
- Engine-version mismatch is unguarded: LM Studio's UI will keep reporting the original version
  number and release notes for a folder whose contents you replaced.
