# Perplexity A/B: Q8_0 vs Q4_0_ROCMFP4 on wikitext-2-raw (wiki.test.raw), same base model
# (Qwen3.6-27B MTP pack in both cases), ROCmFPX runner, ROCm0. Lower PPL = better quality.
#
# This quantifies what the speed A/B (rocmfpx-ab.ps1) deliberately ignores: how much quality the
# 4-bit format actually gives up vs the near-lossless Q8_0. Both files load in the fork runner,
# so the runtime is identical and the only variable is the weight format.
#
# Runtime note: full wiki.test.raw is ~650 chunks of 512; at ~300 t/s prefill that's ~20 min per
# model. Results append to results\rocmfpx-ppl.txt (final PPL line per model).
param([int]$Chunks = 0)   # 0 = full corpus; e.g. -Chunks 200 for a ~3x faster estimate

$ErrorActionPreference = 'Continue'
$root  = Split-Path $PSScriptRoot -Parent
$bin   = Join-Path $root 'bin-rocmfpx\llama-perplexity.exe'
$corp  = Join-Path $root 'data\wikitext-2-raw\wiki.test.raw'
$out   = Join-Path $root 'results\rocmfpx-ppl.txt'

if (-not $env:HIP_PATH) { throw "HIP_PATH not set (HIP SDK required)." }
$env:PATH = "$($env:HIP_PATH.TrimEnd('\'))\bin;$env:PATH"

$models = [ordered]@{
    'Q8_0'         = "$env:USERPROFILE\.lmstudio\models\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q8_0.gguf"
    'ROCmFP4'      = Join-Path $root 'models\Qwen3.6-27B-MTP-ROCmFP4-STRIX-imatrix-embF16-headQ6.gguf'
}

"=== rocmfpx-ppl run $(Get-Date -Format s)  chunks=$(if($Chunks){$Chunks}else{'all'}) ===" | Out-File $out -Append -Encoding utf8

foreach ($name in $models.Keys) {
    $m = $models[$name]
    if (-not (Test-Path $m)) { Write-Host "SKIP $name (missing $m)" -ForegroundColor Red; continue }
    Write-Host "`n============ $name ============" -ForegroundColor Yellow
    $args = @('-m',$m,'-f',$corp,'-dev','ROCm0','-ngl','-1','-fa','on','-t','16')
    if ($Chunks -gt 0) { $args += @('--chunks',"$Chunks") }
    Write-Host "> llama-perplexity $($args -join ' ')" -ForegroundColor DarkGray
    # PPL progress goes to stderr; the "Final estimate" line is what we keep.
    $res = & $bin @args 2>&1 | ForEach-Object { "$_" }
    $final = $res | Where-Object { $_ -match 'Final estimate' } | Select-Object -Last 1
    if (-not $final) { $final = "($name FAILED - last lines: $(($res | Select-Object -Last 3) -join ' | '))" }
    Write-Host $final -ForegroundColor Green
    "$name : $final" | Out-File $out -Append -Encoding utf8
}
Write-Host "`nDONE -> $out" -ForegroundColor Green
