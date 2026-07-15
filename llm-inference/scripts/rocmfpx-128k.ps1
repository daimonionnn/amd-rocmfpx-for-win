# The 128K points for the ROCmFPX question left open in README §8: does ROCmFP4's decode edge
# survive at 128K, where §7 predicts the f16 KV cache (~35 GB) swamps the weight-size advantage?
#
# Two configs, SAME runtime (bin-rocmfpx\, ROCm0) so the only variable is the weight format:
#   fp4-fpx : Q4_0_ROCMFP4_STRIX (15.69 GiB)
#   q8-fpx  : Q8_0 unsloth MTP   (27.04 GiB)
# Per config: pp131072 (prefill, r=1, ~15 min) and tg128 at depth 131072 (-d: llama-bench
# prefills 128K first, then measures decode - a TRUE @128K decode point, r=1, another ~15 min).
# Total ~60 min. Baseline for the lemonade runtime is results\longctx-balanced-128k.csv
# (Performance-bios1.08-iommu-off rows, measured the same day).
$ErrorActionPreference = 'Continue'
$root  = Split-Path $PSScriptRoot -Parent
$bench = Join-Path $root 'bin-rocmfpx\llama-bench.exe'
$out   = Join-Path $root 'results\rocmfpx-128k.csv'

if (-not $env:HIP_PATH) { throw "HIP_PATH not set (HIP SDK required by bin-rocmfpx\)." }
$env:PATH = "$($env:HIP_PATH.TrimEnd('\'))\bin;$env:PATH"

$models = [ordered]@{
    'fp4-fpx' = Join-Path $root 'models\Qwen3.6-27B-MTP-ROCmFP4-STRIX-imatrix-embF16-headQ6.gguf'
    'q8-fpx'  = "$env:USERPROFILE\.lmstudio\models\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q8_0.gguf"
}

if (-not (Test-Path $out)) { 'config,test,avg_ts,stddev_ts,run_time' | Out-File $out -Encoding utf8 }
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

foreach ($name in $models.Keys) {
    $m = $models[$name]
    if (-not (Test-Path $m)) { Write-Host "SKIP $name (missing $m)" -ForegroundColor Red; continue }
    Write-Host "`n============ $name ============" -ForegroundColor Yellow

    $passes = @(
        @{ Label = 'pp131072';       Args = @('-p','131072','-n','0','-ub','1024') }
        @{ Label = 'tg128@d131072';  Args = @('-p','0','-n','128','-d','131072') }
    )
    foreach ($p in $passes) {
        $args = @('-m',$m,'-dev','ROCm0','-ngl','-1','-fa','on','-t','16','-r','1') + $p.Args
        Write-Host "> llama-bench $($args -join ' ')" -ForegroundColor DarkGray
        $csv = & $bench @args -o csv 2>$null
        if ($LASTEXITCODE -ne 0) { Write-Host "  ($name/$($p.Label) FAILED)" -ForegroundColor Red; continue }
        $lines = $csv -split "`r?`n" | Where-Object { $_ -ne '' }
        $r = $lines | Select-Object -Skip 1 | ConvertFrom-Csv -Header ($lines[0] -split ',') | Select-Object -First 1
        Write-Host ("  {0,-15} {1,8:N2} t/s" -f $p.Label, [double]$r.avg_ts) -ForegroundColor White
        "$name,$($p.Label),$($r.avg_ts),$($r.stddev_ts),$stamp" | Out-File $out -Append -Encoding utf8
    }
}
Write-Host "`nDONE -> $out" -ForegroundColor Green
