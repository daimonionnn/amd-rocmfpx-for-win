<#
.SYNOPSIS
    Downloads a llama.cpp Windows build (ROCm 7, HIP, CUDA, Vulkan, CPU) into bin\.
    Idempotent: skips if bin\llama-bench.exe already exists (use -Force to redo).

.PARAMETER Build
    llama.cpp release tag. Default b9910.

.PARAMETER Backend
    Backend package to install.
    - rocm7-gfx1151 (DEFAULT, recommended for this Strix Halo box): ROCm 7 build compiled
      specifically for gfx1151 from lemonade-sdk/llamacpp-rocm. Up to ~4.7x faster long-context
      prefill than the generic HIP build (measured, see ..\llm-inference\README.md). Downloads the
      latest release automatically; the $Build tag is ignored for this backend.
    - hip: official ggml-org generic multi-arch HIP/Radeon build (older ROCm; kept as control).
    - cuda13, cuda12: official CUDA builds — for running this harness on NVIDIA machines.
    - vulkan, cpu: vendor-neutral backends.

.PARAMETER Force
    Re-download and overwrite even if bin\ is populated.
#>
param(
    [string]$Build = 'b9910',
        [ValidateSet('rocm7-gfx1151','hip','cuda13','cuda12','vulkan','cpu')]
        [string]$Backend = 'rocm7-gfx1151',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$BinDir = Join-Path $Root 'bin'
$DlDir  = Join-Path $Root 'dl'
$Bench  = Join-Path $BinDir 'llama-bench.exe'

New-Item -ItemType Directory -Force -Path $BinDir, $DlDir, (Join-Path $Root 'results') | Out-Null

if ((Test-Path $Bench) -and -not $Force) {
    Write-Host "llama-bench.exe already present at $Bench" -ForegroundColor Green
    Write-Host "Use -Force to re-download." -ForegroundColor DarkGray
    & $Bench --list-devices
    return
}

$base = "https://github.com/ggml-org/llama.cpp/releases/download/$Build"

switch ($Backend) {
    'rocm7-gfx1151' {
        # Strix Halo (gfx1151) ROCm 7 build. Fetches the latest lemonade-sdk release so we
        # always get current gfx1151 kernels. This is a different repo/tag scheme than the
        # ggml-org builds, so $base and $assets are (re)derived from the GitHub API here.
        $api = 'https://api.github.com/repos/lemonade-sdk/llamacpp-rocm/releases/latest'
        $rel = Invoke-RestMethod -Uri $api -Headers @{ 'User-Agent' = 'llm-bench-setup' }
        $asset = $rel.assets | Where-Object { $_.name -match 'windows-rocm-gfx1151-x64\.zip$' } | Select-Object -First 1
        if (-not $asset) { throw "No windows-rocm-gfx1151 asset in lemonade-sdk/llamacpp-rocm latest release." }
        $base = "https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/$($rel.tag_name)"
        $assets = @($asset.name)
        $backendLabel = "ROCm 7 (gfx1151 / Strix Halo, lemonade-sdk $($rel.tag_name))"
    }
    'hip' {
        $assets = @(
            "llama-$Build-bin-win-hip-radeon-x64.zip"
        )
        $backendLabel = 'HIP (AMD Radeon / ROCm-style Windows build)'
    }
    'cuda13' {
        $assets = @(
            "llama-$Build-bin-win-cuda-13.3-x64.zip",
            "cudart-llama-bin-win-cuda-13.3-x64.zip"
        )
        $backendLabel = 'CUDA 13.3'
    }
    'cuda12' {
        $assets = @(
            "llama-$Build-bin-win-cuda-12.4-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip"
        )
        $backendLabel = 'CUDA 12.4'
    }
    'vulkan' {
        $assets = @(
            "llama-$Build-bin-win-vulkan-x64.zip"
        )
        $backendLabel = 'Vulkan'
    }
    'cpu' {
        $assets = @(
            "llama-$Build-bin-win-cpu-x64.zip"
        )
        $backendLabel = 'CPU'
    }
}

Write-Host "Installing llama.cpp backend: $backendLabel" -ForegroundColor Cyan

foreach ($z in $assets) {
    $dest = Join-Path $DlDir $z
    Write-Host "Downloading $z ..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri "$base/$z" -OutFile $dest
    Write-Host "Extracting $z -> bin\" -ForegroundColor Cyan
    Expand-Archive -Path $dest -DestinationPath $BinDir -Force
}

if (Test-Path $Bench) {
    Write-Host "`nSetup complete." -ForegroundColor Green
    & $Bench --list-devices
} else {
    throw "Setup finished but llama-bench.exe not found - check the release asset names for tag $Build and backend $Backend."
}
