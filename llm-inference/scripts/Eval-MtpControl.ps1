<#
.SYNOPSIS
  Does MTP speculative decoding change what the model actually answers?

.DESCRIPTION
  Every eval arm in S13 was served with --spec-type draft-mtp. In theory greedy speculative
  decoding is output-equivalent to plain decoding, so that should not matter. In practice S10
  measured that mere floating-point divergence between kernel implementations is enough to flip
  scenarios, and a draft-and-verify path is a different arithmetic route to the same tokens.

  This matters beyond tidiness: quant publishers other than unsloth (mradermacher, bartowski)
  do not ship MTP variants of this model at all. If MTP is output-neutral their files can be
  compared against S13 directly; if it is not, the whole matrix has to be re-run without it
  before anyone else's quants can join.

  Same model, same runtime, same seed, same context - MTP on versus off. Any scenario that
  changes status is MTP's doing.

.EXAMPLE
  .\Eval-MtpControl.ps1
  .\Eval-MtpControl.ps1 -ModelKey q4km
#>
param(
    [ValidateSet('q8','q4km','fp4')][string]$ModelKey = 'q8',
    [int]$Ctx = 32768,
    [int]$Port = 8081
)
$ErrorActionPreference = 'Continue'

$lm   = "$env:USERPROFILE\.lmstudio\models"
$root = Split-Path $PSScriptRoot -Parent
$teb  = Get-ChildItem "$env:LOCALAPPDATA\Temp\claude" -Recurse -Filter 'tool-eval-bench.exe' -EA SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
if (-not $teb) { throw "tool-eval-bench.exe not found." }
$outDir = Join-Path (Split-Path (Split-Path (Split-Path $teb -Parent) -Parent) -Parent) 'teb-out'

$models = @{
    q8   = @{ label='Q8_0';    path="$lm\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q8_0.gguf" }
    q4km = @{ label='Q4_K_M';  path="$lm\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q4_K_M.gguf" }
    fp4  = @{ label='ROCmFP4'; path="$root\models\Qwen3.6-27B-MTP-ROCmFP4-STRIX-imatrix-embF16-headQ6.gguf" }
}
$m = $models[$ModelKey]

foreach ($mode in @('mtp','nomtp')) {
    Write-Host "`n================ $($m.label) / $mode ================" -ForegroundColor Cyan
    Get-Process llama-server -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
    Start-Sleep -Seconds 5

    # Serve-Qwen.ps1 always enables MTP, so the no-MTP arm is launched directly.
    $env:PATH = "$($env:HIP_PATH.TrimEnd('\'))\bin;$env:PATH"
    $srv  = "$root\bin-rocmfpx\llama-server.exe"
    $args = @('-m',$m.path,'-dev','ROCm0','-ngl','-1','-fa','on','-c',"$Ctx",'-t','16',
              '--host','127.0.0.1','--port',"$Port")
    if ($mode -eq 'mtp') { $args += @('--spec-type','draft-mtp','--spec-draft-n-max','6') }

    $p = Start-Process $srv -PassThru -WindowStyle Hidden -ArgumentList $args

    # /health answers 503 with {"status":"loading model"} while loading - wait for a real ok.
    $deadline = (Get-Date).AddMinutes(12); $ready = $false
    while ((Get-Date) -lt $deadline) {
        try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 4 -EA Stop).status -eq 'ok') { $ready = $true; break } } catch { }
        Start-Sleep -Seconds 4
    }
    if (-not $ready) { Write-Host "  server never became ready" -ForegroundColor Red; continue }

    $env:TOOL_EVAL_BASE_URL = "http://127.0.0.1:$Port"
    & $teb --hardmode --seed 42 --no-live --label "$($m.label)-$mode" `
           --json-file "$outDir\mtpctl-$ModelKey-$mode.json" --output-dir $outDir 2>&1 |
        Select-String 'final_score|error' | ForEach-Object { "  $($_.Line)" }

    Stop-Process -Id $p.Id -Force -EA SilentlyContinue
    Start-Sleep -Seconds 5
}

Write-Host "`nCompare with:" -ForegroundColor Green
Write-Host "  .\Compare-ToolEval.ps1 -Baseline $outDir\mtpctl-$ModelKey-mtp.json -Candidate $outDir\mtpctl-$ModelKey-nomtp.json"
