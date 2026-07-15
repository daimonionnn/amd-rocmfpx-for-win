# Q8 + MTP on long context (the user's real target config), via llama-cli speculative mode.
# For each context length: Q8+MTP vs Q8 plain -> shows MTP's decode speedup at long ctx.
# Prompt = natural prose (War & Peace) so MTP acceptance is realistic, not inflated by repetition.
$ErrorActionPreference = 'Continue'
$bin  = 'c:\development\ai-tools-for-win\llm-bench\bin\llama-cli.exe'
$model= "$env:USERPROFILE\.lmstudio\models\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-UD-Q8_K_XL.gguf"
$sp   = 'C:\Users\matt\AppData\Local\Temp\claude\c--development-ai-tools-for-win\4d6c6f75-de03-4360-b40b-5d2c76574ddc\scratchpad'
$out  = "$PSScriptRoot\..\results\longctx-q8-mtp.csv"
if (Test-Path $out) { Remove-Item $out }
'ctx_label,mode,prompt_tokens,prefill_tps,decode_tps' | Out-File $out -Encoding utf8

# (label, prompt file, -c) ; prompt tokens ~34.5k and ~137k
$cases = @(
    @{ label='32k';  file="$sp\prompt-32k.txt";  ctx=36000  },
    @{ label='128k'; file="$sp\prompt-128k.txt"; ctx=138000 }
)

function RunOne($label, $mode, $file, $ctx) {
    $args = @('-m',$model,'-f',$file,'-n','128','-c',"$ctx",'-t','16','-ngl','-1','-fa','on',
              '-dev','ROCm0','--no-warmup','--simple-io','--single-turn','--no-display-prompt','--keep','0')
    if ($mode -eq 'mtp') { $args += @('--spec-type','draft-mtp','--spec-draft-n-max','4') }
    Write-Host "`n======== $label / $mode ========" -ForegroundColor Yellow
    Write-Host "> llama-cli ... $($mode)$(if($mode -eq 'mtp'){' (draft-mtp n=4)'})" -ForegroundColor DarkGray
    $log = "$sp\mtp-$label-$mode.log"
    & $bin @args *>$log
    $txt = Get-Content $log
    $pe = ($txt | Select-String 'prompt eval time' | Select-Object -Last 1)
    $ev = ($txt | Select-String '^eval time'       | Select-Object -Last 1)
    $ptoks = if ($pe -and $pe -match '/\s*(\d+)\s+tokens') { $Matches[1] } else { '?' }
    $ptps  = if ($pe -and $pe -match '([0-9.]+)\s+tokens per second') { [double]$Matches[1] } else { [double]::NaN }
    $dtps  = if ($ev -and $ev -match '([0-9.]+)\s+tokens per second') { [double]$Matches[1] } else { [double]::NaN }
    $summary = ($txt | Select-String '\[\s*Prompt:.*Generation:' | Select-Object -Last 1)
    Write-Host ("  prompt tokens : {0}" -f $ptoks)
    Write-Host ("  prefill t/s   : {0:N2}" -f $ptps) -ForegroundColor White
    Write-Host ("  decode  t/s   : {0:N2}" -f $dtps) -ForegroundColor Green
    if ($summary) { Write-Host "  $($summary.Line.Trim())" -ForegroundColor DarkGray }
    "{0},{1},{2},{3:N2},{4:N2}" -f $label,$mode,$ptoks,$ptps,$dtps | Out-File $out -Append -Encoding utf8
}

foreach ($c in $cases) {
    RunOne $c.label 'plain' $c.file $c.ctx
    RunOne $c.label 'mtp'   $c.file $c.ctx
}
Write-Host "`nDONE -> $out" -ForegroundColor Green
