# The §8 open experiment: is FP4+MTP the fastest 128K decode on this box?
# Head-to-head at ~128K prompt, same prompt file, llama-cli speculative mode (n-max 4):
#   fp4-fork-mtp : ROCmFP4 model on the ROCmFPX fork runtime (the challenger)
#   q8-lemonade-mtp : Q8_0 unsloth MTP on the production ROCm 7 build (the champion, §5: ~9.9 t/s)
# Raw (no-MTP) 128K reference points already exist: FP4 9.19 / Q8 6.06 (results\rocmfpx-128k.csv).
#
# Prompt = natural prose built from wikitext-2 train (data\, fetched by rocmfpx-ppl.ps1's corpus
# download). NOTE: §5 used a different prose source (War & Peace), so cross-compare THIS pair
# internally; treat §5 numbers as approximate reference.
$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
$out  = Join-Path $root 'results\rocmfpx-fp4-mtp-128k.csv'

if (-not $env:HIP_PATH) { throw "HIP_PATH not set." }
$env:PATH = "$($env:HIP_PATH.TrimEnd('\'))\bin;$env:PATH"

# --- build the ~137K-token prompt (~600 KB of prose) once ---------------------
$corpus = Join-Path $root 'data\wikitext-2-raw\wiki.train.raw'
if (-not (Test-Path $corpus)) { throw "wikitext corpus missing - run scripts\rocmfpx-ppl.ps1 once (it downloads it)." }
$prompt = Join-Path $root 'data\prompt-128k.txt'
if (-not (Test-Path $prompt)) {
    $bytes = [IO.File]::ReadAllBytes($corpus)[0..599999]
    [IO.File]::WriteAllBytes($prompt, $bytes)
    Write-Host "Built prompt file: $prompt ($([math]::Round((Get-Item $prompt).Length/1KB)) KB)"
}

$configs = [ordered]@{
    'fp4-fork-mtp' = @{
        Cli   = Join-Path $root 'bin-rocmfpx\llama-cli.exe'
        Model = Join-Path $root 'models\Qwen3.6-27B-MTP-ROCmFP4-STRIX-imatrix-embF16-headQ6.gguf'
    }
    'q8-lemonade-mtp' = @{
        Cli   = 'c:\development\ai-tools-for-win\llm-bench\bin\llama-cli.exe'
        Model = "$env:USERPROFILE\.lmstudio\models\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q8_0.gguf"
    }
}

if (-not (Test-Path $out)) { 'config,prompt_tokens,prefill_tps,decode_tps,run_time' | Out-File $out -Encoding utf8 }
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

foreach ($name in $configs.Keys) {
    $c = $configs[$name]
    if (-not (Test-Path $c.Model)) { Write-Host "SKIP $name (missing model)" -ForegroundColor Red; continue }
    Write-Host "`n======== $name ========" -ForegroundColor Yellow
    $log = Join-Path $root "results\fp4-mtp-128k-$name.log"
    $args = @('-m',$c.Model,'-f',$prompt,'-n','256','-c','138000','-t','16','-ngl','-1','-fa','on',
              '-dev','ROCm0','--no-warmup','--simple-io','--single-turn','--no-display-prompt','--keep','0',
              '--spec-type','draft-mtp','--spec-draft-n-max','4')
    Write-Host "> llama-cli -f prompt-128k.txt -n 256 --spec-type draft-mtp ..." -ForegroundColor DarkGray
    & $c.Cli @args *>$log
    $txt = Get-Content $log
    $pe = ($txt | Select-String 'prompt eval time' | Select-Object -Last 1)
    $ev = ($txt | Select-String '^eval time'       | Select-Object -Last 1)
    $ptoks = if ($pe -and $pe -match '/\s*(\d+)\s+tokens') { $Matches[1] } else { '?' }
    $ptps  = if ($pe -and $pe -match '([0-9.]+)\s+tokens per second') { [double]$Matches[1] } else { [double]::NaN }
    $dtps  = if ($ev -and $ev -match '([0-9.]+)\s+tokens per second') { [double]$Matches[1] } else { [double]::NaN }
    # Fallback: some builds only print the "[ Prompt: X t/s | Generation: Y t/s ]" summary.
    $summary = ($txt | Select-String '\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]' | Select-Object -Last 1)
    if ($summary -and $summary.Line -match 'Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)') {
        if ([double]::IsNaN($ptps)) { $ptps = [double]$Matches[1] }
        if ([double]::IsNaN($dtps)) { $dtps = [double]$Matches[2] }
    }
    Write-Host ("  prompt tokens : {0}"     -f $ptoks)
    Write-Host ("  prefill t/s   : {0:N2}"  -f $ptps) -ForegroundColor White
    Write-Host ("  decode  t/s   : {0:N2}"  -f $dtps) -ForegroundColor Green
    "{0},{1},{2:N2},{3:N2},{4}" -f $name,$ptoks,$ptps,$dtps,$stamp | Out-File $out -Append -Encoding utf8
}
Write-Host "`nDONE -> $out" -ForegroundColor Green
