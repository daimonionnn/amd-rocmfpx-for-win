<#
.SYNOPSIS
  Run tool-eval-bench across several models unattended, one server at a time.

.DESCRIPTION
  Each arm needs its own llama-server, and a 84-scenario run takes ~45 min, so comparing six
  models by hand is a long babysitting job. This starts a server, waits for it to be genuinely
  ready, runs the suite, tears the server down and moves on.

  All arms run on the SAME runtime (the fork, which the ROCmFPX files require) because S10
  measured that swapping runtimes moves results as much as quantization does. Holding it constant
  is what makes the arms comparable.

  Readiness check matters more than it looks: llama.cpp answers /health with
  {"status":"loading model"} and HTTP 503 while it loads, so a naive "did it respond" probe passes
  far too early and the benchmark dies with a 503. Wait for exactly {"status":"ok"}.

.PARAMETER Models
  Keys to run. Default is every model that has a peer worth comparing against.

.PARAMETER Ctx
  Context size for the server. Keep it identical across arms - it is part of the comparison.

.EXAMPLE
  .\Run-EvalQueue.ps1
  .\Run-EvalQueue.ps1 -Models fp6,q6k
#>
param(
    [string[]]$Models = @('q4xl','fp6','q6k','q6xl'),
    [int]$Ctx = 32768,
    [int]$Port = 8081,
    # Quants without MTP heads (mradermacher, bartowski) can only be compared against arms that
    # also ran without it - S13's MTP control measured 3 scenarios genuinely shifting.
    [switch]$NoMtp,
    # Without MTP the model decodes 2.3x slower and scenarios start blowing the default 120 s
    # limit. Those get excluded from scoring while still reporting status "fail", which reads as a
    # quality regression and is not one. Raise this whenever -NoMtp is used.
    [int]$Timeout = 0,
    [string]$Suffix = ''
)
$ErrorActionPreference = 'Continue'

$lm   = "$env:USERPROFILE\.lmstudio\models"
$root = Split-Path $PSScriptRoot -Parent
$teb  = Get-ChildItem "$env:LOCALAPPDATA\Temp\claude" -Recurse -Filter 'tool-eval-bench.exe' -EA SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
if (-not $teb) { throw "tool-eval-bench.exe not found - install it into a venv first." }
$outDir = Split-Path (Split-Path $teb -Parent) -Parent | Split-Path -Parent
$outDir = Join-Path $outDir 'teb-out'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$catalog = @{
    fp4  = @{ label='ROCmFP4';        path="$root\models\Qwen3.6-27B-MTP-ROCmFP4-STRIX-imatrix-embF16-headQ6.gguf" }
    q4km = @{ label='Q4_K_M';         path="$lm\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q4_K_M.gguf" }
    q4xl = @{ label='UD-Q4_K_XL';     path="$lm\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-UD-Q4_K_XL.gguf" }
    fp6  = @{ label='Chadrockv2-FP6'; path="$lm\jcbtc\Chadrockv2-Qwen3.6-27B-ROCmFP6-STRIX-QUALITY\Chadrockv2-Qwen3.6-27B-ROCmFP6-STRIX-QUALITY.gguf" }
    q6k  = @{ label='Q6_K';           path="$lm\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q6_K.gguf" }
    q6xl = @{ label='UD-Q6_K_XL';     path="$lm\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-UD-Q6_K_XL.gguf" }
    q8   = @{ label='Q8_0';           path="$lm\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q8_0.gguf" }
    # A community fine-tune rather than a quant of the base: multi-stage merge, abliterated
    # ("heretic"/"uncensored"), claiming ARC-C 711 and beating base Qwen3.6-27B on 6 of 7
    # benchmarks. Reasoning scores and agent discipline are different capabilities, and
    # abliteration removes the refusal directions our evals specifically probe.
    dau  = @{ label='DavidAU-711';    path="$lm\DavidAU\Fable-Fusion-711-MTP-Q4_K_M.gguf" }
    # Same weights as unsloth's, different builder - the format-vs-builder control. No MTP heads.
    mrad = @{ label='mradermacher-Q4_K_M';    path="$lm\mradermacher\Qwen3.6-27B.Q4_K_M.gguf" }
    mradi1 = @{ label='mradermacher-i1-Q4_K_M'; path="$lm\mradermacher\Qwen3.6-27B.i1-Q4_K_M.gguf" }
}

function Stop-Servers {
    Get-Process llama-server -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
    Start-Sleep -Seconds 5
}

foreach ($key in $Models) {
    $m = $catalog[$key]
    if (-not $m) { Write-Host "unknown model key '$key' - skipping" -ForegroundColor Yellow; continue }
    if (-not (Test-Path $m.path)) { Write-Host "missing file for '$key' - skipping" -ForegroundColor Yellow; continue }

    Write-Host "`n================ $($m.label) ================" -ForegroundColor Cyan
    Stop-Servers

    if ($NoMtp) {
        # Serve-Qwen.ps1 always enables MTP, so the no-MTP arm launches the fork binary directly.
        $env:PATH = "$($env:HIP_PATH.TrimEnd('\'))\bin;$env:PATH"
        $serve = Start-Process "$root\bin-rocmfpx\llama-server.exe" -PassThru -WindowStyle Hidden -ArgumentList @(
            '-m',$m.path,'-dev','ROCm0','-ngl','-1','-fa','on','-c',"$Ctx",'-t','16',
            '--host','127.0.0.1','--port',"$Port")
    } else {
        $serve = Start-Process powershell -PassThru -WindowStyle Hidden -ArgumentList @(
            '-NoProfile','-File',"$root\Serve-Qwen.ps1",
            '-Runtime','rocmfpx','-Model',$m.path,'-Ctx',"$Ctx",'-Port',"$Port",'-ListenAddress','127.0.0.1')
    }

    # Poll for a real ready state, not merely a reachable socket.
    $deadline = (Get-Date).AddMinutes(12)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        try {
            $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 4 -EA Stop
            if ($h.status -eq 'ok') { $ready = $true; break }
        } catch { }
        Start-Sleep -Seconds 4
    }
    if (-not $ready) { Write-Host "  server never became ready - skipping $($m.label)" -ForegroundColor Red; continue }
    Write-Host "  server ready, running suite..." -ForegroundColor DarkGray

    $env:TOOL_EVAL_BASE_URL = "http://127.0.0.1:$Port"
    $tag = if ($Suffix) { $Suffix } elseif ($NoMtp) { '-nomtp' } else { '-fork' }
    $a = @('--hardmode','--seed','42','--no-live','--label',"$($m.label)$tag",
           '--json-file',"$outDir\$key$tag.json",'--output-dir',$outDir)
    if ($Timeout -gt 0) { $a += @('--timeout',"$Timeout") }
    & $teb @a 2>&1 | Select-String 'final_score|Score:|error' | ForEach-Object { "  $($_.Line)" }

    Stop-Servers
}

Write-Host "`nQUEUE DONE. Results in $outDir" -ForegroundColor Green
