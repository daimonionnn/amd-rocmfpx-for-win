# Steps 2+3: REAL target model Qwen3.6-27B *Q8_0* on the ROCm 7 gfx1151 build (now bin\),
# comparing f16 KV cache vs Q8 KV cache across the long-context curve.
# Deliverables: (a) real Q8-model TTFT, (b) the Q8-KV-cache effect on prefill + memory.
$ErrorActionPreference = 'Continue'
$bin   = 'c:\development\ai-tools-for-win\llm-bench\bin\llama-bench.exe'   # bin\ is now ROCm 7
$model = "$env:USERPROFILE\.lmstudio\models\lmstudio-community\Qwen3.6-27B-GGUF\Qwen3.6-27B-Q8_0.gguf"
$out   = "$PSScriptRoot\..\results\longctx-q8-kv.csv"
if (Test-Path $out) { Remove-Item $out }
if (-not (Test-Path $model)) { throw "Q8 model not found: $model" }

$sizes = 16384,32768,65536,131072   # r1: prefill throughput is near-deterministic at these lengths

function RunArm($label, $kvArgs) {
    foreach ($s in $sizes) {
        Write-Host "`n============ $label  pp$s ============" -ForegroundColor Yellow
        $args = @('-m',$model,'-ngl','-1','-fa','on','-p',"$s",'-n','0','-t','16','-r','1','-ub','1024') + $kvArgs
        Write-Host "> llama-bench $($args -join ' ')" -ForegroundColor DarkGray
        $md = & $bin @args -o md 2>$null
        $code = $LASTEXITCODE
        $md | ForEach-Object { Write-Host $_ }
        if ($code -ne 0) { Write-Host "  ($label pp$s FAILED / likely OOM, code $code)" -ForegroundColor Red; continue }
        $csv = & $bin @args -o csv 2>$null
        $lines = $csv -split "`r?`n" | Where-Object { $_ -ne '' }
        if (-not (Test-Path $out)) { "kv,$($lines[0])" | Out-File $out -Encoding utf8 }
        foreach ($d in ($lines | Select-Object -Skip 1)) { "$label,$d" | Out-File $out -Append -Encoding utf8 }
    }
}

RunArm 'f16-kv' @()
RunArm 'q8-kv'  @('-ctk','q8_0','-ctv','q8_0')

Write-Host "`nDONE -> $out" -ForegroundColor Green
