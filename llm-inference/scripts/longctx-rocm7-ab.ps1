# A/B: ROCm 7 gfx1151-specific build (lemonade-sdk/llamacpp-rocm b1295) vs the
# official multi-arch b9910 HIP build, on the long-context prefill curve.
# Control numbers (b9910) already in results\longctx-prefill.csv.
$ErrorActionPreference = 'Continue'
$bin  = 'c:\development\ai-tools-for-win\llm-bench\bin-rocm7-gfx1151\llama-bench.exe'
$qwen = "$env:USERPROFILE\.lmstudio\models\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-UD-Q4_K_XL.gguf"
$out  = "$PSScriptRoot\..\results\longctx-prefill-rocm7.csv"
if (Test-Path $out) { Remove-Item $out }

# Fast-signal subset (skip 131072 until a win shows). Matched settings to the control.
$sizes = 4096,16384,32768,65536
foreach ($s in $sizes) {
    Write-Host "`n==================== rocm7 pp$s ====================" -ForegroundColor Yellow
    $args = @('-m',$qwen,'-ngl','-1','-fa','on','-p',"$s",'-n','0','-t','16','-r','2','-ub','1024')
    Write-Host "> llama-bench $($args -join ' ')" -ForegroundColor DarkGray
    $md = & $bin @args -o md 2>$null
    $code = $LASTEXITCODE
    $md | ForEach-Object { Write-Host $_ }
    if ($code -ne 0) { Write-Host "  (rocm7 pp$s FAILED, code $code)" -ForegroundColor Red; continue }
    $csv = & $bin @args -o csv 2>$null
    $lines = $csv -split "`r?`n" | Where-Object { $_ -ne '' }
    if (-not (Test-Path $out)) { $lines[0] | Out-File $out -Encoding utf8 }
    foreach ($d in ($lines | Select-Object -Skip 1)) { $d | Out-File $out -Append -Encoding utf8 }
}
Write-Host "`nDONE. CSV -> $out" -ForegroundColor Green
