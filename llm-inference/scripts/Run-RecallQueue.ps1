<#
.SYNOPSIS
  Run the long-context recall test across several models, one server at a time.

.DESCRIPTION
  Wraps Test-LongContextRecall.ps1 with the server lifecycle. Every arm gets the identical
  haystack, needles, depths and seed, so the only variable is the weights.

  Includes Q8_K_XL alongside Q8_0 deliberately. S16 measured ~3.6 points of variation between
  BUILDERS of the same quant; two different 8-bit recipes from the SAME builder measure something
  else - how much the recipe alone moves the result. If those two disagree, "Q8" is not a fixed
  reference point either, and every "vs Q8" figure in this repo inherits that spread.

.EXAMPLE
  .\Run-RecallQueue.ps1
  .\Run-RecallQueue.ps1 -Models q8,q4km
#>
param(
    [string[]]$Models = @('q8','q4km','q8xl'),
    [int]$Ctx = 110000,
    [int]$HaystackTokens = 100000,
    [int]$Port = 8081
)
$ErrorActionPreference = 'Continue'

$lm   = "$env:USERPROFILE\.lmstudio\models\unsloth\Qwen3.6-27B-MTP-GGUF"
$root = Split-Path $PSScriptRoot -Parent
$out  = "$root\results\longctx-recall.csv"

$catalog = @{
    q8    = @{ label='Q8_0';       path="$lm\Qwen3.6-27B-Q8_0.gguf" }
    q8xl  = @{ label='UD-Q8_K_XL'; path="$lm\Qwen3.6-27B-UD-Q8_K_XL.gguf" }
    q4km  = @{ label='Q4_K_M';     path="$lm\Qwen3.6-27B-Q4_K_M.gguf" }
    q6k   = @{ label='Q6_K';       path="$lm\Qwen3.6-27B-Q6_K.gguf" }
    q4xl  = @{ label='UD-Q4_K_XL'; path="$lm\Qwen3.6-27B-UD-Q4_K_XL.gguf" }
    fp4   = @{ label='ROCmFP4';    path="$root\models\Qwen3.6-27B-MTP-ROCmFP4-STRIX-imatrix-embF16-headQ6.gguf" }
}

if (Test-Path $out) { Remove-Item $out }
if (-not $env:HIP_PATH) { throw "HIP_PATH is not set." }
$env:PATH = "$($env:HIP_PATH.TrimEnd('\'))\bin;$env:PATH"

foreach ($key in $Models) {
    $m = $catalog[$key]
    if (-not $m -or -not (Test-Path $m.path)) { Write-Host "skipping '$key'" -ForegroundColor Yellow; continue }

    Get-Process llama-server -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
    Start-Sleep -Seconds 6

    # --parallel 1: a single client, and every extra slot reserves KV it will never use.
    $p = Start-Process "$root\bin-rocmfpx\llama-server.exe" -PassThru -WindowStyle Hidden -ArgumentList @(
        '-m',$m.path,'-dev','ROCm0','-ngl','-1','-fa','on','-c',"$Ctx",'-t','16','--parallel','1',
        '--spec-type','draft-mtp','--spec-draft-n-max','6','--host','127.0.0.1','--port',"$Port")

    $deadline = (Get-Date).AddMinutes(15); $ready = $false
    while ((Get-Date) -lt $deadline) {
        try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 4 -EA Stop).status -eq 'ok') { $ready = $true; break } } catch { }
        Start-Sleep -Seconds 5
    }
    if (-not $ready) { Write-Host "  $($m.label): server never became ready" -ForegroundColor Red; continue }

    & "$PSScriptRoot\Test-LongContextRecall.ps1" -Label $m.label -Port $Port -HaystackTokens $HaystackTokens -OutFile $out

    Stop-Process -Id $p.Id -Force -EA SilentlyContinue
    Get-Process llama-server -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
    Start-Sleep -Seconds 6
}

Write-Host "`n=== summary ===" -ForegroundColor Cyan
Import-Csv $out | Group-Object label | ForEach-Object {
    $hit = @($_.Group | Where-Object found -eq 'True').Count
    "{0,-14} {1,2}/{2,2} = {3,5:P0}" -f $_.Name, $hit, $_.Count, ($hit/$_.Count)
}
Write-Host "`nMisses by depth (a lossy quant should fail deep first, not at random):" -ForegroundColor Cyan
Import-Csv $out | Where-Object found -ne 'True' |
    Select-Object label,needle,depth,expected | Sort-Object label,depth | Format-Table -AutoSize
