<#
.SYNOPSIS
  128k long-context prefill + generation benchmark for Qwen3.6-27B Q4_K_XL and Q8_0.
  Recorded on BIOS Balanced profile.

.DESCRIPTION
  For each model runs:
    1. pp131072 + tg128 combined (r=1)  — prefill at 128k tokens, then 128 gen tokens
    2. tg128 fresh (r=3)                — generation from an empty KV cache

  Results appended to results\longctx-balanced-128k.csv.

.NOTES
  Expected runtime: ~65-70 min per model for the pp131072 pass (~2.5 h total).
#>
param(
    [int]$GenTokens    = 128,
    [int]$Threads      = 16,
    [string]$Device    = 'auto',
    [string]$BiosProfile = 'Balanced'
)

$ErrorActionPreference = 'Continue'

$bench = 'C:\development\ai-tools-for-win\llm-bench\bin\llama-bench.exe'
$out   = "$PSScriptRoot\..\results\longctx-balanced-128k.csv"

$models = [ordered]@{
    'Q4_K_XL' = "$env:USERPROFILE\.lmstudio\models\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-UD-Q4_K_XL.gguf"
    'Q8_0'    = "$env:USERPROFILE\.lmstudio\models\lmstudio-community\Qwen3.6-27B-GGUF\Qwen3.6-27B-Q8_0.gguf"
}

if (-not (Test-Path $out)) {
    'bios_profile,quant,test,n_prompt,n_gen,avg_ts,stddev_ts,run_time' | Out-File $out -Encoding utf8
}

$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

foreach ($quant in $models.Keys) {
    $m = $models[$quant]
    Write-Host ""
    Write-Host "=================================================================" -ForegroundColor Cyan
    Write-Host "  Model  : $quant ($([math]::Round((Get-Item $m).Length/1GB,1)) GiB)" -ForegroundColor Cyan
    Write-Host "  Profile: $BiosProfile" -ForegroundColor Cyan
    Write-Host "  When   : $stamp" -ForegroundColor Cyan
    Write-Host "=================================================================" -ForegroundColor Cyan

    # --- 1) 128k prefill + tg128 combined (single model load) -----------------
    Write-Host "`n[1/2] pp131072 + tg$GenTokens  (r=1 — expected ~65-70 min)" -ForegroundColor Yellow
    $args1 = @('-m',$m,'-ngl','-1','-dev',$Device,'-fa','on',
               '-p','131072','-n',"$GenTokens",'-t',"$Threads",'-r','1','-ub','1024')
    Write-Host "  > llama-bench $($args1 -join ' ')" -ForegroundColor DarkGray
    $csv1 = & $bench @args1 -o csv 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
    } else {
        $lines1 = $csv1 -split "`r?`n" | Where-Object { $_ -ne '' }
        $rows1  = $lines1 | Select-Object -Skip 1 | ConvertFrom-Csv -Header ($lines1[0] -split ',')
        foreach ($r in $rows1) {
            $test = if ([int]$r.n_gen -gt 0) { "tg$($r.n_gen)@128k" } else { "pp$($r.n_prompt)" }
            Write-Host ("  {0,-20} {1,10:N2} +/- {2,6:N2} t/s" -f $test,[double]$r.avg_ts,[double]$r.stddev_ts) -ForegroundColor White
            "$BiosProfile,$quant,$test,$($r.n_prompt),$($r.n_gen),$($r.avg_ts),$($r.stddev_ts),$stamp" |
                Out-File $out -Append -Encoding utf8
        }
    }

    # --- 2) Fresh tg128 (no pre-filled KV cache) -------------------------
    Write-Host "`n[2/2] tg$GenTokens fresh  (r=3)" -ForegroundColor Yellow
    $args2 = @('-m',$m,'-ngl','-1','-dev',$Device,'-fa','on',
               '-p','0','-n',"$GenTokens",'-t',"$Threads",'-r','3')
    Write-Host "  > llama-bench $($args2 -join ' ')" -ForegroundColor DarkGray
    $csv2 = & $bench @args2 -o csv 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
    } else {
        $lines2 = $csv2 -split "`r?`n" | Where-Object { $_ -ne '' }
        $rows2  = $lines2 | Select-Object -Skip 1 | ConvertFrom-Csv -Header ($lines2[0] -split ',')
        foreach ($r in $rows2) {
            Write-Host ("  tg{0} fresh   {1,10:N2} +/- {2,6:N2} t/s" -f $r.n_gen,[double]$r.avg_ts,[double]$r.stddev_ts) -ForegroundColor White
            "$BiosProfile,$quant,tg$($r.n_gen)-fresh,$($r.n_prompt),$($r.n_gen),$($r.avg_ts),$($r.stddev_ts),$stamp" |
                Out-File $out -Append -Encoding utf8
        }
    }
}

Write-Host ""
Write-Host "DONE. Results -> $out" -ForegroundColor Green
