# Prefill tuning sweep for Strix Halo (gfx1151) ROCm llama.cpp.
# Question: does forcing hipBLASLt or micro-batch tuning help SHORT prefill?
# Answer (b9910): no -- all within noise. See ..\README.md.
$ErrorActionPreference = 'Continue'

$repoRoot = Split-Path $PSScriptRoot -Parent   # ...\llm-inference
$devRoot  = Split-Path $repoRoot -Parent       # ...\ai-tools-for-win
$bin = Join-Path $devRoot 'llm-bench\bin\llama-bench.exe'
if (-not (Test-Path $bin)) { throw "llama-bench.exe not found at $bin (run ..\llm-bench\Setup.ps1)." }

$gemma = "$env:USERPROFILE\.lmstudio\models\lmstudio-community\gemma-3-27B-it-qat-GGUF\gemma-3-27B-it-QAT-Q4_0.gguf"
$qwen  = "$env:USERPROFILE\.lmstudio\models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf"
$out   = "$PSScriptRoot\..\results\prefill-tune.csv"
if (Test-Path $out) { Remove-Item $out }

function Run($label, $envval, $model, $benchArgs) {
    Write-Host "`n==================== $label ====================" -ForegroundColor Yellow
    if ($null -eq $envval) {
        Remove-Item Env:ROCBLAS_USE_HIPBLASLT -ErrorAction SilentlyContinue
    } else {
        $env:ROCBLAS_USE_HIPBLASLT = $envval
    }
    Write-Host "ROCBLAS_USE_HIPBLASLT=$envval  > llama-bench $($benchArgs -join ' ')" -ForegroundColor DarkGray
    $md = & $bin @benchArgs -m $model -o md 2>$null
    $md | ForEach-Object { Write-Host $_ }
    $csv = & $bin @benchArgs -m $model -o csv 2>$null
    $lines = $csv -split "`r?`n" | Where-Object { $_ -ne '' }
    if (-not (Test-Path $out)) { "label,$($lines[0])" | Out-File $out -Encoding utf8 }
    foreach ($d in ($lines | Select-Object -Skip 1)) { "$label,$d" | Out-File $out -Append -Encoding utf8 }
}

$base = @('-ngl','-1','-fa','on','-p','512,2048','-n','0','-t','16','-r','3')
Run 'gemma-default'  $null $gemma $base
Run 'gemma-lt-on'    '1'   $gemma $base
Run 'gemma-lt-off'   '0'   $gemma $base
Run 'gemma-ubsweep-default' $null $gemma @('-ngl','-1','-fa','on','-p','2048','-n','0','-t','16','-r','3','-ub','256,512,1024','-b','2048')
Run 'qwen-default' $null $qwen $base
Run 'qwen-lt-on'   '1'   $qwen $base

Write-Host "`nDONE. CSV -> $out" -ForegroundColor Green
