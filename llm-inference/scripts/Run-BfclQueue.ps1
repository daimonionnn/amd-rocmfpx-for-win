<#
.SYNOPSIS
  Second-opinion tool-calling eval: run BFCL across several quants of the same model.

.DESCRIPTION
  S13 ranked seven quants on tool-eval-bench's 84 scenarios and found the ordering does not track
  bit-width. Before acting on that - it would change this repo's production recommendation - the
  result needs a scenario set nobody here authored. BFCL is that set: independent, established,
  and specifically about function calling.

  Three things about this pairing have to be stated up front.

  BFCL has 175 model handlers and no generic OpenAI-compatible one, so a Qwen3.6-27B served over
  llama.cpp has to borrow `Qwen/Qwen3-32B-FC` - a different model generation. That handler decides
  how tools are rendered and how replies are parsed, so our ABSOLUTE scores are meaningless and
  not comparable to the public leaderboard. The comparison between our own arms stays valid
  because every arm carries the identical handicap.

  Results are written to result/<handler-name>/, which is the same path for every arm. Left alone
  they overwrite each other. Each arm's output is moved aside before the next one starts.

  Roughly 1.5 h per category per model, so scope deliberately: one category across all arms
  answers more than four categories on one arm.

.PARAMETER Models
  Arm keys to run. Default is the three that decide the open question from S13.

.PARAMETER Categories
  BFCL test categories. `irrelevance` maps onto the Restraint & Refusal behaviour where S13 saw
  differences; `multi_turn_miss_func` is the closest thing to the real agent loop.

.EXAMPLE
  .\Run-BfclQueue.ps1
  .\Run-BfclQueue.ps1 -Models q8,q4km,fp4 -Categories irrelevance,multi_turn_miss_func
#>
param(
    [string[]]$Models = @('q8','q4km','fp4'),
    [string[]]$Categories = @('irrelevance'),
    [int]$Ctx = 32768,
    [int]$Port = 8081
)
$ErrorActionPreference = 'Continue'

$lm     = "$env:USERPROFILE\.lmstudio\models"
$root   = Split-Path $PSScriptRoot -Parent
$scratch = Get-ChildItem "$env:LOCALAPPDATA\Temp\claude" -Recurse -Directory -Filter 'bfcl-venv' -EA SilentlyContinue |
           Select-Object -First 1 -ExpandProperty FullName
if (-not $scratch) { throw "bfcl-venv not found." }
$bfcl   = "$scratch\Scripts\bfcl.exe"
$pkgDir = "$scratch\Lib\site-packages"
$outRoot = Join-Path (Split-Path $scratch -Parent) 'bfcl-out'
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

$HANDLER = 'Qwen/Qwen3-32B-FC'
$HANDLER_DIR = 'Qwen_Qwen3-32B-FC'

$catalog = @{
    q8   = @{ label='Q8_0';       path="$lm\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q8_0.gguf" }
    q4km = @{ label='Q4_K_M';     path="$lm\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q4_K_M.gguf" }
    q4xl = @{ label='UD-Q4_K_XL'; path="$lm\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-UD-Q4_K_XL.gguf" }
    q6k  = @{ label='Q6_K';       path="$lm\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q6_K.gguf" }
    fp4  = @{ label='ROCmFP4';    path="$root\models\Qwen3.6-27B-MTP-ROCmFP4-STRIX-imatrix-embF16-headQ6.gguf" }
    fp6  = @{ label='Chadrockv2-FP6'; path="$lm\jcbtc\Chadrockv2-Qwen3.6-27B-ROCmFP6-STRIX-QUALITY\Chadrockv2-Qwen3.6-27B-ROCmFP6-STRIX-QUALITY.gguf" }
}

# BFCL prints emoji; the Windows console default (cp1252) cannot encode them and the run dies
# during scoring, after the expensive generation step has already succeeded.
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
# The endpoint variable takes a bare host - BFCL appends the scheme and /v1 itself. Passing a URL
# yields http://http://... and it polls /models forever without ever explaining why.
$env:LOCAL_SERVER_ENDPOINT = '127.0.0.1'
$env:LOCAL_SERVER_PORT = "$Port"
$env:REMOTE_OPENAI_BASE_URL = "http://127.0.0.1:$Port/v1"
$env:REMOTE_OPENAI_API_KEY = 'EMPTY'

foreach ($key in $Models) {
    $m = $catalog[$key]
    if (-not $m -or -not (Test-Path $m.path)) { Write-Host "skipping '$key'" -ForegroundColor Yellow; continue }

    Write-Host "`n================ $($m.label) ================" -ForegroundColor Cyan
    Get-Process llama-server -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
    Start-Sleep -Seconds 5

    $srv = Start-Process powershell -PassThru -WindowStyle Hidden -ArgumentList @(
        '-NoProfile','-File',"$root\Serve-Qwen.ps1",
        '-Runtime','rocmfpx','-Model',$m.path,'-Ctx',"$Ctx",'-Port',"$Port",'-ListenAddress','127.0.0.1')

    $deadline = (Get-Date).AddMinutes(12); $ready = $false
    while ((Get-Date) -lt $deadline) {
        try { if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 4 -EA Stop).status -eq 'ok') { $ready = $true; break } } catch { }
        Start-Sleep -Seconds 4
    }
    if (-not $ready) { Write-Host "  server never became ready" -ForegroundColor Red; continue }

    foreach ($cat in $Categories) {
        Write-Host "  --- $cat ---" -ForegroundColor DarkGray
        Push-Location $pkgDir
        & $bfcl generate --model $HANDLER --test-category $cat --backend vllm --skip-server-setup --num-threads 1 2>&1 |
            Select-String -Pattern '100%\|' | Select-Object -Last 1 | ForEach-Object { "    $($_.Line.Trim())" }
        & $bfcl evaluate --model $HANDLER --test-category $cat 2>&1 |
            Select-String -Pattern 'Accuracy' | ForEach-Object { "    $($_.Line.Trim())" }
        Pop-Location
    }

    # Same handler for every arm means the same output path for every arm. Move, do not merge.
    $dest = Join-Path $outRoot $key
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    foreach ($d in 'result','score') {
        $src = Join-Path $pkgDir "$d\$HANDLER_DIR"
        if (Test-Path $src) { Move-Item $src (Join-Path $dest $d) -Force }
    }
    Get-ChildItem "$pkgDir\score" -Filter '*.csv' -EA SilentlyContinue | Move-Item -Destination $dest -Force

    Stop-Process -Id $srv.Id -Force -EA SilentlyContinue
    Get-Process llama-server -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
    Start-Sleep -Seconds 5
}

Write-Host "`nQUEUE DONE -> $outRoot" -ForegroundColor Green
Get-ChildItem $outRoot -Directory | ForEach-Object { "  $($_.Name)" }
