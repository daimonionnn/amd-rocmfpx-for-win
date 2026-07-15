# Long-context prefill curve for the REAL workload: Qwen3.6-27B, 100K+ tokens.
# Measures TTFT-relevant prefill throughput as context grows (the O(S^2) wall).
$ErrorActionPreference = 'Continue'

$repoRoot = Split-Path $PSScriptRoot -Parent
$devRoot  = Split-Path $repoRoot -Parent
$bin = Join-Path $devRoot 'llm-bench\bin\llama-bench.exe'
if (-not (Test-Path $bin)) { throw "llama-bench.exe not found at $bin (run ..\llm-bench\Setup.ps1)." }

# Closest available to the user's model (Qwen3.6-27B). Q4_K_XL; Q8 would be ~30-40% slower.
$qwen = "$env:USERPROFILE\.lmstudio\models\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-UD-Q4_K_XL.gguf"
$out  = "$PSScriptRoot\..\results\longctx-prefill.csv"
if (Test-Path $out) { Remove-Item $out }

# Prefill only (-n 0), fa on, ub1024 (marginal best from sweep). Ascending so
# partial results survive if the largest OOMs. Add a Q8-KV arm for comparison.
$sizes = 4096,16384,32768,65536,131072
foreach ($s in $sizes) {
    Write-Host "`n==================== pp$s ====================" -ForegroundColor Yellow
    $args = @('-m',$qwen,'-ngl','-1','-fa','on','-p',"$s",'-n','0','-t','16','-r','2','-ub','1024')
    Write-Host "> llama-bench $($args -join ' ')" -ForegroundColor DarkGray
    $md = & $bin @args -o md 2>$null
    $code = $LASTEXITCODE
    $md | ForEach-Object { Write-Host $_ }
    if ($code -ne 0) { Write-Host "  (pp$s FAILED / likely OOM, code $code)" -ForegroundColor Red; continue }
    $csv = & $bin @args -o csv 2>$null
    $lines = $csv -split "`r?`n" | Where-Object { $_ -ne '' }
    if (-not (Test-Path $out)) { $lines[0] | Out-File $out -Encoding utf8 }
    foreach ($d in ($lines | Select-Object -Skip 1)) { $d | Out-File $out -Append -Encoding utf8 }
}
Write-Host "`nDONE. CSV -> $out" -ForegroundColor Green
