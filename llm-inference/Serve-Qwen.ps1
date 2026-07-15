<#
.SYNOPSIS
  Launch Qwen3.6-27B with MTP speculative decoding as an OpenAI-compatible API server
  on the ROCm 7 gfx1151 build, reachable from other PCs on the LAN.

.DESCRIPTION
  This is the "final recommended config" from .\README.md:
    ROCm 7 build + Qwen3.6-27B-Q8_K_XL + MTP (draft-mtp, n-max 4) + f16 KV.
  Point your Hermes agent (on another PC) at:  http://<this-PC-LAN-IP>:<Port>/v1

.PARAMETER Runtime
  Which llama.cpp build to serve with.
    rocm7   (DEFAULT) ..\llm-bench\bin\ - lemonade ROCm 7 gfx1151 build (provisioned by
                      ..\llm-bench\Setup.ps1), standard GGUF quants.
    rocmfpx           bin-rocmfpx\ - the ROCmFPX fork (Setup-ROCmFPX.ps1). REQUIRED for
                      ROCmFP4/ROCmFPX-format models; it also still reads standard GGUFs.
  Each runtime has its own default -Model; -Device defaults to ROCm0 for both (the rocmfpx
  build also exposes Vulkan0, which upstream claims is its stronger decode path - A/B it).

.PARAMETER Port
  TCP port to listen on. Default 8081 — NOT 8080, which is permanently occupied by the
  AgentService process on this machine (binding it fails with "couldn't bind HTTP server socket").

.PARAMETER ListenAddress
  Bind address. Default 0.0.0.0 (all interfaces, so other PCs can reach it).
  Use 127.0.0.1 to restrict to this machine only.

.PARAMETER Ctx
  Context window (prompt + generation). Default 262144 = Qwen3.6-27B's full native
  training length (n_ctx_train; no rope scaling needed). Costs ~70 GB of f16 KV cache
  pre-allocation at this size — fine on this 128 GB box (~97 GB total with the Q8 model),
  lower it on smaller machines. Going beyond 262144 would need '--rope-scaling yarn'.

.PARAMETER DraftNMax
  MTP max draft tokens. Default 4 (measured best/LM-Studio default).

.PARAMETER ApiKey
  Optional API key. If set, clients must send 'Authorization: Bearer <key>'.
  Recommended if the port is reachable beyond a trusted LAN.

.PARAMETER Model
  Override the GGUF path. Default depends on -Runtime (see below).

.PARAMETER Device
  llama.cpp device. Default ROCm0. With -Runtime rocmfpx, Vulkan0 is also available.

.EXAMPLE
  .\Serve-Qwen.ps1
  .\Serve-Qwen.ps1 -Port 9000 -Ctx 163840 -ApiKey "my-secret"
  .\Serve-Qwen.ps1 -Runtime rocmfpx
  .\Serve-Qwen.ps1 -Runtime rocmfpx -Device Vulkan0
#>
param(
    [ValidateSet('rocm7','rocmfpx')]
    [string]$Runtime = 'rocm7',
    [int]$Port = 8081,
    [string]$ListenAddress = '0.0.0.0',
    [int]$Ctx = 262144,
    [int]$DraftNMax = 4,
    [string]$ApiKey = '',
    [string]$Model = '',
    [string]$Device = 'ROCm0'
)

$ErrorActionPreference = 'Stop'

switch ($Runtime) {
    'rocm7' {
        $BinDir = Join-Path $PSScriptRoot '..\llm-bench\bin'
        # Q8_0 (26.6 GB) = same file/quant LM Studio runs; near-identical quality to Q8_K_XL but
        # ~1.25-1.4x faster decode (decode is memory-bandwidth bound). For max quality use
        # -Model ...\Qwen3.6-27B-UD-Q8_K_XL.gguf ; for max speed ...\Qwen3.6-27B-UD-Q4_K_XL.gguf
        $defaultModel = "$env:USERPROFILE\.lmstudio\models\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q8_0.gguf"
        $setupHint    = '..\llm-bench\Setup.ps1'
        $label        = 'ROCm 7 / gfx1151 (lemonade)'
    }
    'rocmfpx' {
        $BinDir = Join-Path $PSScriptRoot 'bin-rocmfpx'
        $defaultModel = Join-Path $PSScriptRoot 'models\Qwen3.6-27B-MTP-ROCmFP4-STRIX-imatrix-embF16-headQ6.gguf'
        $setupHint    = '.\Setup-ROCmFPX.ps1  (then .\Get-ROCmFPXModel.ps1)'
        $label        = 'ROCmFPX fork / gfx1151'
        # The fork's binaries are staged without the HIP runtime (rocblas/hipblaslt Tensile
        # libraries are several GB), so they load it from the HIP SDK.
        if (-not $env:HIP_PATH) { throw "HIP_PATH is not set - the ROCmFPX build needs the HIP SDK at runtime." }
        $env:PATH = "$($env:HIP_PATH.TrimEnd('\'))\bin;$env:PATH"
    }
}

$Server = Join-Path $BinDir 'llama-server.exe'
if (-not $Model) { $Model = $defaultModel }

if (-not (Test-Path $Server)) { throw "llama-server.exe not found at $Server  (run $setupHint)" }
if (-not (Test-Path $Model))  { throw "Model not found:`n  $Model" }

# Build args: ROCm 7 + Q8_K_XL + MTP (draft-mtp) + f16 KV, all layers on GPU, flash-attn on.
$args = @(
    '-m',    $Model,
    '-dev',  $Device,
    '-ngl',  '-1',
    '-fa',   'on',
    '-c',    "$Ctx",
    '-t',    '16',
    '--spec-type',      'draft-mtp',
    '--spec-draft-n-max', "$DraftNMax",
    '--host', $ListenAddress,
    '--port', "$Port"
    # --jinja is enabled by default (needed for tool-calling); f16 KV is the default.
)
if ($ApiKey) { $args += @('--api-key', $ApiKey) }

# Show the endpoint(s) the Hermes agent should point at.
$lanIps = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' } |
    Select-Object -ExpandProperty IPAddress)

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Qwen3.6-27B + MTP  ($label)" -ForegroundColor Cyan
Write-Host " Model   : $([IO.Path]::GetFileName($Model))" -ForegroundColor Gray
Write-Host " Runtime : $Runtime -> $BinDir    Device: $Device" -ForegroundColor Gray
Write-Host " Context : $Ctx    Draft n-max: $DraftNMax    KV: f16" -ForegroundColor Gray
Write-Host " Auth    : $(if ($ApiKey) { 'API key REQUIRED' } else { 'NONE (open on the network)' })" -ForegroundColor $(if ($ApiKey) {'Gray'} else {'Yellow'})
Write-Host "------------------------------------------------------------------" -ForegroundColor Cyan
Write-Host " Point Hermes (OpenAI base_url) at one of:" -ForegroundColor Green
if ($ListenAddress -eq '0.0.0.0') {
    foreach ($ip in $lanIps) { Write-Host "     http://${ip}:$Port/v1" -ForegroundColor Green }
}
Write-Host "     http://localhost:$Port/v1   (this machine)" -ForegroundColor Green
Write-Host " Chat endpoint: POST /v1/chat/completions   |   Web UI: http://localhost:$Port" -ForegroundColor DarkGray
Write-Host "==================================================================" -ForegroundColor Cyan
if ($ListenAddress -eq '0.0.0.0') {
    Write-Host " NOTE: if another PC can't connect, allow the port through Windows Firewall:" -ForegroundColor Yellow
    Write-Host "   New-NetFirewallRule -DisplayName 'llama-server $Port' -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port" -ForegroundColor DarkGray
}
if (-not $ApiKey) {
    Write-Host " NOTE: no API key set - anyone on the network can use this endpoint. Pass -ApiKey to lock it." -ForegroundColor Yellow
}
Write-Host ""
Write-Host "> llama-server $($args -join ' ')" -ForegroundColor DarkGray
Write-Host ""

& $Server @args
