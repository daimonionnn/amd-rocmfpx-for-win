# Q4_K_XL + MTP: prefill and decode at ~32K and ~137K, via llama-cli speculative mode.
# Same method as longctx-q8-mtp.ps1 so numbers compare directly. Reads t/s from
# llama.cpp's own "[ Prompt: X t/s | Generation: Y t/s ]" summary line.
$ErrorActionPreference = 'Continue'
$bin  = 'c:\development\ai-tools-for-win\llm-bench\bin\llama-cli.exe'
$model= "$env:USERPROFILE\.lmstudio\models\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-UD-Q4_K_XL.gguf"
$sp   = 'C:\Users\matt\AppData\Local\Temp\claude\c--development-ai-tools-for-win\4d6c6f75-de03-4360-b40b-5d2c76574ddc\scratchpad'
$out  = "$PSScriptRoot\..\results\longctx-q4-mtp.csv"
if (Test-Path $out) { Remove-Item $out }
'ctx_label,prefill_tps,decode_tps' | Out-File $out -Encoding utf8

$cases = @(
    @{ label='32k';  file="$sp\prompt-32k.txt";  ctx=36000  },
    @{ label='128k'; file="$sp\prompt-128k.txt"; ctx=138000 }
)

foreach ($c in $cases) {
    Write-Host "`n======== Q4_K_XL / $($c.label) (MTP) ========" -ForegroundColor Yellow
    $args = @('-m',$model,'-f',$c.file,'-n','128','-c',"$($c.ctx)",'-t','16','-ngl','-1','-fa','on',
              '-dev','ROCm0','--no-warmup','--simple-io','--single-turn','--no-display-prompt','--keep','0',
              '--spec-type','draft-mtp','--spec-draft-n-max','4')
    $log = "$sp\q4-$($c.label).log"
    & $bin @args *>$log
    $summary = (Get-Content $log | Select-String '\[\s*Prompt:.*Generation:' | Select-Object -Last 1)
    $pp = $null; $tg = $null
    if ($summary -and $summary.Line -match 'Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s') {
        $pp = [double]$Matches[1]; $tg = [double]$Matches[2]
    }
    Write-Host ("  prefill t/s : {0}" -f $(if($pp){'{0:N1}' -f $pp}else{'?'})) -ForegroundColor White
    Write-Host ("  decode  t/s : {0}" -f $(if($tg){'{0:N2}' -f $tg}else{'?'})) -ForegroundColor Green
    "{0},{1:N1},{2:N2}" -f $c.label,$pp,$tg | Out-File $out -Append -Encoding utf8
}
Write-Host "`nDONE -> $out" -ForegroundColor Green
