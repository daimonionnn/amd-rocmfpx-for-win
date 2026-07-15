<#
.SYNOPSIS
  One-shot launcher for the production-quality model on the ROCmFPX fork runtime:
  Qwen3.6-27B **Q8_0 + MTP** served from bin-rocmfpx\ (with the chat WebUI) on port 8081.

.DESCRIPTION
  Thin wrapper over Serve-Qwen.ps1 that pins the combination we otherwise have to spell out:
    -Runtime rocmfpx   the fork runtime (chat WebUI on the same port)
    -Model  Q8_0 MTP   the quality-first production quant (README.md par.5) - NOT the fork's
                       default ROCmFP4 model.
  Any llama-server already running is stopped first (one model at a time on this box).

  MEASURED CAVEAT (2026-07-15): with MTP the fork generates SLOWER than the lemonade build on
  this same model (16.3 vs 20.9 t/s TG, identical prompt, temp 0). Its draft-mtp path
  under-speculates: 93% draft acceptance but far fewer drafts issued (1381 vs 2119 per 2000
  tokens), so less of the output rides the speculative fast path. This more than offsets the
  fork's +5-7% prefill edge. For plain Q8 serving prefer .\Serve-Qwen.ps1; this script mainly
  remains useful for A/B and for the fork's WebUI. Details: README.md par.8.

.PARAMETER Port
  Default 8081 (8080 is taken by AgentService on this machine).

.PARAMETER ApiKey
  Optional; forwarded to Serve-Qwen.ps1.

.EXAMPLE
  .\Serve-Q8-Fork.ps1
  .\Serve-Q8-Fork.ps1 -Port 9000 -ApiKey "my-secret"
#>
param(
    [int]$Port = 8081,
    [string]$ApiKey = '',
    [int]$Ctx = 131072
)

$ErrorActionPreference = 'Stop'

$Q8 = "$env:USERPROFILE\.lmstudio\models\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q8_0.gguf"

$running = Get-Process llama-server -ErrorAction SilentlyContinue
if ($running) {
    Write-Host "Stopping running llama-server (PID $($running.Id -join ', '))..." -ForegroundColor Yellow
    $running | Stop-Process -Force
    Start-Sleep -Seconds 2
}

$serveArgs = @{
    Runtime = 'rocmfpx'
    Model   = $Q8
    Port    = $Port
    Ctx     = $Ctx
}
if ($ApiKey) { $serveArgs.ApiKey = $ApiKey }

& (Join-Path $PSScriptRoot 'Serve-Qwen.ps1') @serveArgs
