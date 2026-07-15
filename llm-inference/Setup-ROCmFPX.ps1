<#
.SYNOPSIS
    Builds the ROCmFPX llama.cpp fork (charlie12345/ROCmFPX) from source into bin-rocmfpx\.

.DESCRIPTION
    ROCmFPX adds AMD-specific GGUF weight formats (Q4_0_ROCMFP4, Q6_0_ROCMFPX, Q8_0_ROCMFPX, ...)
    that STOCK llama.cpp CANNOT LOAD - those models need this runner. Upstream ships no releases
    and no Windows build script (only bash), so this script is the Windows equivalent of
    scripts\build-strix-rocmfp4-mtp.sh, using the same cmake flags.

    Builds BOTH backends in one go (like the upstream strix script):
      -dev ROCm0    HIP/ROCm path   (matches our existing ROCm 7 stack)
      -dev Vulkan0  Vulkan path     (upstream calls this the primary decode path on Strix Halo)
    so both can be A/B'd against each other.

    Requirements (all verified present on this box):
      - AMD HIP SDK  (HIP_PATH, ships gfx1151 kernels)      C:\Program Files\AMD\ROCm\7.1
      - Vulkan SDK   (VULKAN_SDK, for glslc)                C:\VulkanSDK\1.4.341.1
      - Visual Studio with the C++ workload (MSVC host toolchain for ROCm's clang)
      - cmake + ninja + git

    The HIP runtime DLLs (rocblas/hipblaslt + their Tensile kernel libraries, several GB) are NOT
    copied into bin-rocmfpx\ - they are loaded straight from the HIP SDK by prepending
    $env:HIP_PATH\bin to PATH at launch. Serve-Qwen.ps1 -Runtime rocmfpx does this for you.

.PARAMETER Arch
    GPU target. Default gfx1151 (Strix Halo / Radeon 8060S).

.PARAMETER Tune
    ROCmFPX decode tuning profile (upstream scripts\rocmfp4-decode-tune-flags.sh).
    'stable' (default) adds no extra flags. Others (e.g. rocmfpx-strix-nwarps2,
    strix-moe-rpb2) inject a -DGGML_ROCMFP*_... define to A/B decode kernel shapes.

.PARAMETER MsvcVersion
    MSVC toolset to build against. Default 14.44.
    Do NOT bump this blindly: MSVC 14.51 (VS 2026's default) added a _CLANG_BUILTIN block to
    <cmath> that declares isgreater/isless/isunordered/... as builtin overloads. ROCm clang's
    __clang_cuda_math_forward_declares.h then redeclares them __device__, and every .cu in
    ggml-cuda\ fails with "__device__ function 'isgreater' cannot overload __host__ __device__
    function 'isgreater'". 14.44 has no such block and compiles clean.

.PARAMETER Jobs
    Parallel compile jobs. Default = CPU count.

.PARAMETER Update
    git pull the existing clone before building (default: build the clone as-is).

.PARAMETER Force
    Reconfigure/rebuild from scratch even if bin-rocmfpx\llama-cli.exe already exists.

.EXAMPLE
    .\Setup-ROCmFPX.ps1
    .\Setup-ROCmFPX.ps1 -Update -Force
    .\Setup-ROCmFPX.ps1 -Tune rocmfpx-strix-nwarps2 -BuildDir build-win-nwarps2
#>
param(
    [string]$Arch  = 'gfx1151',
    [string]$Tune  = 'stable',
    [string]$MsvcVersion = '14.44',
    [int]   $Jobs  = 0,
    [string]$BuildDir = 'build-win-rocmfpx',
    [switch]$Update,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$Root    = $PSScriptRoot
$SrcDir  = Join-Path $Root 'src\ROCmFPX'
$BinDir  = Join-Path $Root 'bin-rocmfpx'
$Repo    = 'https://github.com/charlie12345/ROCmFPX.git'
$Cli     = Join-Path $BinDir 'llama-cli.exe'

if ($Jobs -le 0) { $Jobs = [Environment]::ProcessorCount }

if ((Test-Path $Cli) -and -not $Force) {
    Write-Host "ROCmFPX already built: $Cli" -ForegroundColor Green
    Write-Host "Use -Force to rebuild." -ForegroundColor DarkGray
    return
}

# ---------- prerequisites ----------
function Require($name, $value, $hint) {
    if (-not $value) { throw "$name not found. $hint" }
    return $value
}

$hipPath = Require 'HIP SDK ($env:HIP_PATH)' $env:HIP_PATH `
    'Install the AMD HIP SDK for Windows (https://rocm.docs.amd.com/projects/install-on-windows).'
$vkSdk   = Require 'Vulkan SDK ($env:VULKAN_SDK)' $env:VULKAN_SDK `
    'Install the Vulkan SDK (https://vulkan.lunarg.com/sdk/home#windows).'
Require 'cmake' (Get-Command cmake -ErrorAction SilentlyContinue) 'winget install Kitware.CMake' | Out-Null
Require 'ninja' (Get-Command ninja -ErrorAction SilentlyContinue) 'winget install Ninja-build.Ninja' | Out-Null
Require 'git'   (Get-Command git   -ErrorAction SilentlyContinue) 'winget install Git.Git' | Out-Null

$hipPath = $hipPath.TrimEnd('\')
if (-not (Test-Path (Join-Path $hipPath 'bin\clang++.exe'))) {
    throw "HIP SDK at $hipPath has no bin\clang++.exe - the HIP SDK install looks incomplete."
}

# ROCm's clang is a Windows/MSVC-ABI compiler: it needs the MSVC headers+libs in the environment,
# so the whole build runs inside a VS x64 developer shell (vcvars64.bat).
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$vsPath  = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw "No Visual Studio with the C++ workload (MSVC x64 tools) found." }
$vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

# Pin the toolset (see .PARAMETER MsvcVersion - 14.51's <cmath> breaks ROCm clang's HIP headers).
$toolsets = Get-ChildItem (Join-Path $vsPath 'VC\Tools\MSVC') -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty Name
$toolset = $toolsets | Where-Object { $_.StartsWith($MsvcVersion) } | Sort-Object | Select-Object -Last 1
if (-not $toolset) {
    throw @"
MSVC toolset $MsvcVersion not installed. Installed: $($toolsets -join ', ')
Install it via the VS Installer ('MSVC v143 - VS 2022 C++ x64/x86 build tools'), or pass
-MsvcVersion <one of the above> - but note 14.5x+ is known to fail the HIP compile.
"@
}
Write-Host "MSVC toolset: $toolset (pinned via -vcvars_ver=$MsvcVersion)" -ForegroundColor DarkGray

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Building ROCmFPX llama.cpp fork" -ForegroundColor Cyan
Write-Host "  arch    : $Arch     tune profile: $Tune     jobs: $Jobs"        -ForegroundColor Gray
Write-Host "  HIP SDK : $hipPath"                                              -ForegroundColor Gray
Write-Host "  Vulkan  : $vkSdk"                                                -ForegroundColor Gray
Write-Host "  MSVC    : $vsPath"                                               -ForegroundColor Gray
Write-Host "==================================================================" -ForegroundColor Cyan

# ---------- source ----------
if (-not (Test-Path $SrcDir)) {
    Write-Host "Cloning $Repo ..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path (Split-Path $SrcDir -Parent) | Out-Null
    git clone --depth 1 $Repo $SrcDir
    if ($LASTEXITCODE -ne 0) { throw "git clone failed." }
} elseif ($Update) {
    Write-Host "Updating clone ..." -ForegroundColor Cyan
    git -C $SrcDir pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw "git pull failed." }
}

# ---------- decode tuning profile -> extra HIP flags ----------
# Mirrors scripts\rocmfp4-decode-tune-flags.sh; 'stable' = no extra flags.
$tuneFlags = switch ($Tune) {
    'stable'                  { '' }
    'strix-moe-rpb1'          { '-DGGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK=1' }
    'strix-moe-rpb2'          { '-DGGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK=2' }
    'strix-moe-rpb3'          { '-DGGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK=3' }
    'strix-moe-rpb4'          { '-DGGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK=4' }
    'strix-nwarps1'           { '-DGGML_ROCMFP4_RDNA35_NWARPS=1' }
    'strix-nwarps2'           { '-DGGML_ROCMFP4_RDNA35_NWARPS=2' }
    'strix-nwarps4'           { '-DGGML_ROCMFP4_RDNA35_NWARPS=4' }
    'strix-mmid3'             { '-DGGML_ROCMFP4_RDNA35_MMID_MAX_BATCH=3' }
    'strix-mmid4'             { '-DGGML_ROCMFP4_RDNA35_MMID_MAX_BATCH=4' }
    'rocmfpx-strix-moe-rpb1'  { '-DGGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK=1' }
    'rocmfpx-strix-moe-rpb2'  { '-DGGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK=2' }
    'rocmfpx-strix-moe-rpb3'  { '-DGGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK=3' }
    'rocmfpx-strix-moe-rpb4'  { '-DGGML_ROCMFPX_MOE_MMVQ_ROWS_PER_BLOCK=4' }
    'rocmfpx-strix-nwarps1'   { '-DGGML_ROCMFPX_RDNA35_NWARPS=1' }
    'rocmfpx-strix-nwarps2'   { '-DGGML_ROCMFPX_RDNA35_NWARPS=2' }
    'rocmfpx-strix-nwarps4'   { '-DGGML_ROCMFPX_RDNA35_NWARPS=4' }
    'rocmfpx-strix-rpb2'      { '-DGGML_ROCMFPX_RDNA35_RPB_WIDE=2' }
    'rocmfpx-strix-mmid1'     { '-DGGML_ROCMFPX_RDNA35_MMID_MAX_BATCH=1' }
    'rocmfpx-strix-mmid2'     { '-DGGML_ROCMFPX_RDNA35_MMID_MAX_BATCH=2' }
    'rocmfpx-strix-mmid3'     { '-DGGML_ROCMFPX_RDNA35_MMID_MAX_BATCH=3' }
    'rocmfpx-strix-mmid4'     { '-DGGML_ROCMFPX_RDNA35_MMID_MAX_BATCH=4' }
    'rocmfpx-strix-vdr2'      { '-DGGML_ROCMFP6_Q8_1_MMVQ_VDR=2' }
    'rocmfpx-strix-vdr8'      { '-DGGML_ROCMFP6_Q8_1_MMVQ_VDR=8' }
    default { throw "Unknown tune profile '$Tune'. See scripts\rocmfp4-decode-tune-flags.sh in $SrcDir." }
}

$Build   = Join-Path $SrcDir $BuildDir
if ($Force -and (Test-Path $Build)) {
    Write-Host "Removing stale build dir $Build" -ForegroundColor DarkGray
    Remove-Item -Recurse -Force $Build
}

# ---------- configure + build (inside vcvars64) ----------
# Same flags as scripts\build-strix-rocmfp4-mtp.sh, except WEBUI=ON (upstream builds headless;
# we want llama-server's chat UI - it embeds via npm build, falling back to prebuilt HF assets).
# GGML_HIP_FORCE_MMQ is REQUIRED by the ROCmFP4 kernels; GGML_VULKAN gives the second backend.
$targets = 'llama-cli llama-server llama-bench llama-quantize llama-perplexity test-backend-ops'
$bat = Join-Path $env:TEMP "build-rocmfpx-$PID.bat"
@"
@echo on
call "$vcvars" -vcvars_ver=$MsvcVersion || exit /b 1
set "PATH=$hipPath\bin;%PATH%"
cmake -S "$SrcDir" -B "$Build" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=clang ^
  -DCMAKE_CXX_COMPILER=clang++ ^
  -DGGML_HIP=ON ^
  -DGGML_HIP_FORCE_MMQ=ON ^
  -DGGML_HIP_ROCWMMA_FATTN=OFF ^
  -DGGML_VULKAN=ON ^
  -DGGML_CUDA=OFF ^
  -DCMAKE_HIP_ARCHITECTURES=$Arch ^
  -DGPU_TARGETS=$Arch ^
  -DAMDGPU_TARGETS=$Arch ^
  -DCMAKE_HIP_FLAGS="$tuneFlags" ^
  -DLLAMA_BUILD_SERVER=ON ^
  -DLLAMA_BUILD_WEBUI=ON ^
  -DLLAMA_USE_PREBUILT_WEBUI=ON ^
  -DLLAMA_BUILD_TESTS=ON ^
  -DGGML_BUILD_TESTS=OFF || exit /b 1
cmake --build "$Build" -j $Jobs --target $targets || exit /b 1
"@ | Set-Content -Path $bat -Encoding ascii

try {
    & cmd.exe /c $bat
    if ($LASTEXITCODE -ne 0) { throw "ROCmFPX build failed (exit $LASTEXITCODE). See output above." }
} finally {
    Remove-Item $bat -Force -ErrorAction SilentlyContinue
}

# ---------- stage binaries ----------
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
Copy-Item (Join-Path $Build 'bin\*') $BinDir -Recurse -Force
Write-Host "`nStaged binaries -> $BinDir" -ForegroundColor Green

if (-not (Test-Path $Cli)) { throw "Build finished but $Cli is missing." }

# Devices: HIP runtime lives in the SDK, so put it on PATH for the probe (and at runtime).
$env:PATH = "$hipPath\bin;$env:PATH"
& (Join-Path $BinDir 'llama-bench.exe') --list-devices

Write-Host @"

ROCmFPX is ready.
  Run a ROCmFPX model:  .\Serve-Qwen.ps1 -Runtime rocmfpx -Model <rocmfp4.gguf>
  Get a model:          .\Get-ROCmFPXModel.ps1
  Quantize your own:    bin-rocmfpx\llama-quantize.exe src-BF16.gguf out.gguf Q4_0_ROCMFP4_FAST
  Benchmark vs Q8_0:    .\scripts\rocmfpx-ab.ps1
"@ -ForegroundColor Cyan
