<#
.SYNOPSIS
  ROCm benchmark pack for AMD Strix Halo (APU): Gemma, Qwen MoE, and Qwen MTP.

.DESCRIPTION
  Runs llama.cpp benchmarks on an AMD ROCm-capable build.

  1) Gemma 3 27B (dense) with llama-bench
  2) Qwen3.6 35B-A3B (MoE) with llama-bench
  3) Qwen3.6 27B MTP with llama-cli speculative mode draft-mtp

  Flash Attention is enabled by default. For the MTP model, max draft tokens are
  set to 4 by default (matches LM Studio "Max Draft Tokens = 4").

.PARAMETER Threads
  Comma-separated thread counts for llama-bench runs. Default = physical core count.

.PARAMETER PromptTokens
  Prompt length for all tests. Default 512.

.PARAMETER GenTokens
  Generation length for all tests. Default 128.

.PARAMETER Reps
  Repetitions per test. Default 3.

.PARAMETER GpuLayers
  GPU layers for llama-bench tests. Use -1 for all layers. Default -1.

.PARAMETER Device
  llama.cpp device selector for ROCm backend. Default auto.
  Example values: auto, 0, 0/1

.PARAMETER FlashAttn
  Enable/disable flash attention. Default on.

.PARAMETER MtpMaxDraftTokens
  MTP max draft tokens for Qwen MTP model. Default 4.

.PARAMETER GemmaModelPath
  Override Gemma model path.

.PARAMETER SkipGemma
  Skip the Gemma benchmark entirely. Useful when the Gemma model is not installed.

.PARAMETER QwenModelPath
  Override Qwen MoE model path.

.PARAMETER QwenMtpModelPath
  Override Qwen MTP model path. Can be a .gguf file or a folder containing .gguf.

.EXAMPLE
  .\Run-RocmTest.ps1
  .\Run-RocmTest.ps1 -Device 0 -GpuLayers -1 -FlashAttn on
#>
param(
    [string]$Threads,
    [int]$PromptTokens = 512,
    [int]$GenTokens = 128,
    [int]$Reps = 3,
    [int]$GpuLayers = -1,
    [string]$Device = 'auto',
    [ValidateSet('on','off','auto')]
    [string]$FlashAttn = 'on',
    [int]$MtpMaxDraftTokens = 4,
    [switch]$SkipGemma,
    [string]$GemmaModelPath,
    [string]$QwenModelPath,
    [string]$QwenMtpModelPath
)

. "$PSScriptRoot\Common.ps1"

Assert-Bench
Assert-Cli

if (-not $GemmaModelPath) { $GemmaModelPath = $GemmaModel }
if (-not $QwenModelPath) { $QwenModelPath = $QwenMoeModel }
if (-not $QwenMtpModelPath) { $QwenMtpModelPath = $QwenMtpModel }

if (-not $SkipGemma) {
  $gemmaResolved = Resolve-ModelPath -PathLike $GemmaModelPath -ModelName 'Gemma 3 27B'
}
$qwenResolved = Resolve-ModelPath -PathLike $QwenModelPath -ModelName 'Qwen3.6 35B-A3B'
$qwenMtpResolved = Resolve-ModelPath -PathLike $QwenMtpModelPath -ModelName 'Qwen3.6 27B MTP'

if (-not $Threads) { $Threads = (Get-CpuInfo).Cores.ToString() }

$cliDevice = Resolve-LlamaDevice -BinaryPath $Cli -RequestedDevice $Device

Write-Host "ROCm benchmark pack (AMD APU / Strix Halo)" -ForegroundColor Yellow
Write-Host "Flash Attention: $FlashAttn" -ForegroundColor Yellow
Write-Host "MTP max draft tokens: $MtpMaxDraftTokens" -ForegroundColor Yellow
Write-Host "Skip Gemma: $SkipGemma" -ForegroundColor Yellow
Write-Host "CLI device: $cliDevice" -ForegroundColor Yellow

# --- 1) Gemma 3 27B (dense) ---------------------------------------------------
if ($SkipGemma) {
  Write-Host "Skipping Gemma benchmark." -ForegroundColor DarkYellow
} else {
  $gemmaArgs = @(
    '-m', $gemmaResolved,
    '-ngl', $GpuLayers,
    '-dev', $Device,
    '-fa', $FlashAttn,
    '-p', $PromptTokens,
    '-n', $GenTokens,
    '-t', $Threads,
    '-r', $Reps
  )
  Invoke-Bench -BenchArgs $gemmaArgs -TestName 'rocm-gemma3-27b'
}

# --- 2) Qwen3.6 35B-A3B (MoE) ------------------------------------------------
$qwenArgs = @(
    '-m', $qwenResolved,
    '-ngl', $GpuLayers,
    '-dev', $Device,
    '-fa', $FlashAttn,
    '-p', $PromptTokens,
    '-n', $GenTokens,
    '-t', $Threads,
    '-r', $Reps
)
Invoke-Bench -BenchArgs $qwenArgs -TestName 'rocm-qwen3.6-35b-a3b'

# --- 3) Qwen3.6 27B MTP -------------------------------------------------------
$mtpArgs = @(
  '-dev', $cliDevice,
    '-ngl', $GpuLayers,
    '-fa', $FlashAttn,
    '--spec-type', 'draft-mtp',
    '--spec-draft-n-max', $MtpMaxDraftTokens
)
Invoke-CliTimingBench `
    -ModelPath $qwenMtpResolved `
    -TestName 'rocm-qwen3.6-27b-mtp' `
    -PromptTokens $PromptTokens `
    -GenTokens $GenTokens `
    -Threads ([int]($Threads -split ',' | Select-Object -First 1)) `
    -Reps $Reps `
    -ExtraArgs $mtpArgs
