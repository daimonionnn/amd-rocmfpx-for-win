<#
.SYNOPSIS
    Downloads a ROCmFPX-format GGUF into models\. These load ONLY in the ROCmFPX runner
    (bin-rocmfpx\, see Setup-ROCmFPX.ps1) - stock llama.cpp and LM Studio cannot read them.

.DESCRIPTION
    Deliberately NOT stored under %USERPROFILE%\.lmstudio\models\ like our other GGUFs: LM Studio
    would index them and fail to load them (unknown tensor types). Default target is models\ here.

    Resumable (curl -C -). A 17-30 GB download; re-run to continue after an interruption.

.PARAMETER Name
    Which model to fetch:
      qwen3.6-27b-fp4  (DEFAULT, 16.9 GB) plunderstruck/Qwen3.6-27B-MTP-ROCmFP4-GGUF
                       Same base model as our Q8_0 default, so it A/Bs cleanly against it.
                       Q4_0_ROCMFP4, imatrix, STRIX preset, embF16/headQ6, MTP heads.
      qwopus-27b-q8    (29.4 GB) philtheriver/Qwopus3.6-27B-Coder-MTP-ROCmFPX, Q8_0_ROCMFPX.
      qwopus-27b-q8-agent (29.9 GB) same repo, Q8_0_ROCMFPX_AGENT - agent-protected preset
                       (tool-calling / JSON coherency). Different base model (coder fine-tune).
      qwopus-27b-q6-agent (26.2 GB) same repo, Q6_0_ROCMFPX_AGENT, headQ6.

.PARAMETER Dest
    Target directory. Default: models\ next to this script.

.EXAMPLE
    .\Get-ROCmFPXModel.ps1
    .\Get-ROCmFPXModel.ps1 -Name qwopus-27b-q8-agent
#>
param(
    [ValidateSet('qwen3.6-27b-fp4','qwopus-27b-q8','qwopus-27b-q8-agent','qwopus-27b-q6-agent')]
    [string]$Name = 'qwen3.6-27b-fp4',
    [string]$Dest = (Join-Path $PSScriptRoot 'models')
)

$ErrorActionPreference = 'Stop'

$catalog = @{
    'qwen3.6-27b-fp4' = @{
        Repo = 'plunderstruck/Qwen3.6-27B-MTP-ROCmFP4-GGUF'
        File = 'Qwen3.6-27B-MTP-ROCmFP4-STRIX-imatrix-embF16-headQ6.gguf'
        Size = '16.9 GB'; Quant = 'Q4_0_ROCMFP4 (imatrix, STRIX)'
    }
    'qwopus-27b-q8' = @{
        Repo = 'philtheriver/Qwopus3.6-27B-Coder-MTP-ROCmFPX'
        File = 'Qwopus3.6-27B-Coder-MTP-STRIX-embF16-Q8_0_ROCMFPX.gguf'
        Size = '29.4 GB'; Quant = 'Q8_0_ROCMFPX'
    }
    'qwopus-27b-q8-agent' = @{
        Repo = 'philtheriver/Qwopus3.6-27B-Coder-MTP-ROCmFPX'
        File = 'Qwopus3.6-27B-Coder-MTP-STRIX-embF16-Q8_0_ROCMFPX_AGENT.gguf'
        Size = '29.9 GB'; Quant = 'Q8_0_ROCMFPX_AGENT'
    }
    'qwopus-27b-q6-agent' = @{
        Repo = 'philtheriver/Qwopus3.6-27B-Coder-MTP-ROCmFPX'
        File = 'Qwopus3.6-27B-Coder-MTP-STRIX-embF16-headQ6-Q6_0_ROCMFPX_AGENT.gguf'
        Size = '26.2 GB'; Quant = 'Q6_0_ROCMFPX_AGENT'
    }
}

$m   = $catalog[$Name]
$url = "https://huggingface.co/$($m.Repo)/resolve/main/$($m.File)?download=true"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$out = Join-Path $Dest $m.File

if (Test-Path $out) {
    $haveGB = [math]::Round((Get-Item $out).Length / 1GB, 1)
    Write-Host "Already present ($haveGB GB): $out" -ForegroundColor Green
    Write-Host "Resuming if incomplete (curl -C -); delete the file to start over." -ForegroundColor DarkGray
}

Write-Host "Downloading $Name  [$($m.Quant), $($m.Size)]" -ForegroundColor Cyan
Write-Host "  from $($m.Repo)" -ForegroundColor DarkGray
Write-Host "  to   $out"       -ForegroundColor DarkGray

& curl.exe -L -C - --fail --retry 5 --retry-delay 5 -o $out $url
if ($LASTEXITCODE -ne 0) { throw "Download failed (curl exit $LASTEXITCODE). Re-run to resume." }

$gb = [math]::Round((Get-Item $out).Length / 1GB, 2)
Write-Host "`nDone: $out  ($gb GB)" -ForegroundColor Green
Write-Host "Serve it:  .\Serve-Qwen.ps1 -Runtime rocmfpx -Model `"$out`"" -ForegroundColor Cyan
