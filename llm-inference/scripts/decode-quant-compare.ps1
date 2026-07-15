# Short decode-speed comparison across quants: Qwen3.6-27B Q4_K_M vs Q6_K vs Q8_0.
# Decode (tg) is the memory-bandwidth-bound regime, so bigger quant = slower generation.
# Measured on the ROCm 7 gfx1151 build (bin\). Two depths: fresh (0) and long-ctx (32768).
$ErrorActionPreference = 'Continue'
$bin  = 'c:\development\ai-tools-for-win\llm-bench\bin\llama-bench.exe'
$dir  = "$env:USERPROFILE\.lmstudio\models\lmstudio-community\Qwen3.6-27B-GGUF"
$out  = "$PSScriptRoot\..\results\decode-quant-compare.csv"
if (Test-Path $out) { Remove-Item $out }

# Ordered smallest -> largest quant (bytes), which is fastest -> slowest for decode.
$models = [ordered]@{
    'Q4_K_M' = "$dir\Qwen3.6-27B-Q4_K_M.gguf"
    'Q6_K'   = "$dir\Qwen3.6-27B-Q6_K.gguf"
    'Q8_0'   = "$dir\Qwen3.6-27B-Q8_0.gguf"
}

foreach ($q in $models.Keys) {
    $m = $models[$q]
    if (-not (Test-Path $m)) { Write-Host "SKIP $q (missing: $m)" -ForegroundColor Red; continue }
    Write-Host "`n============ $q ============" -ForegroundColor Yellow
    # -p 0 -n 128 = pure generation; -d 0,32768 = fresh + after a 32K prefill.
    $args = @('-m',$m,'-ngl','-1','-fa','on','-p','0','-n','128','-d','0,32768','-t','16','-r','3')
    Write-Host "> llama-bench $($args -join ' ')" -ForegroundColor DarkGray
    $md = & $bin @args -o md 2>$null
    $code = $LASTEXITCODE
    $md | ForEach-Object { Write-Host $_ }
    if ($code -ne 0) { Write-Host "  ($q FAILED, code $code)" -ForegroundColor Red; continue }
    $csv = & $bin @args -o csv 2>$null
    $lines = $csv -split "`r?`n" | Where-Object { $_ -ne '' }
    if (-not (Test-Path $out)) { "quant,$($lines[0])" | Out-File $out -Encoding utf8 }
    foreach ($d in ($lines | Select-Object -Skip 1)) { "$q,$d" | Out-File $out -Append -Encoding utf8 }
}
Write-Host "`nDONE -> $out" -ForegroundColor Green
