# MTP draft-depth sweep on the PRODUCTION config (Q8_0 + lemonade ROCm 7 build).
#
# Motivation: the ciru-ai "ROCmFP6 STRIX QUALITY" release serves with --spec-draft-n-max 6
# (+ p-min 0.0) and reports a decode t/s x GiB product ~25% above our best measured MTP point.
# Since our S8 A/B showed ROCmFPX *formats* only buy a few percent over the bandwidth line,
# the suspect is the drafting depth, not the weight format. This sweep tests that on Q8_0.
#
# Axes: --spec-draft-n-max 4 (our default) / 6 (theirs) / 8, plus n-max 6 at LM Studio's p-min 0.75.
# Contexts: fresh (~2K prompt) and ~32K, both from the same wikitext prose corpus, temp 0.
#
#   -Repeat N   run each config N times (default 1). Decode here carries ~+-0.7 t/s of run-to-run
#               variance from draft acceptance, so a single pass cannot resolve a few-percent gap.
#   -Only       restrict to one context label ('2k' or '32k') to spend the reps where it matters.
#   -Runtime    'rocm7' (default, lemonade production build) or 'rocmfpx' (the fork). The fork can
#               only be reached through its own binaries, and S8 measured its MTP path 22% slower
#               than lemonade on an identical Q8_0 - at n-max 4, before we knew depth mattered.
#               Re-run with -Runtime rocmfpx to test whether deeper drafting closes that gap.
param(
    [int]$Repeat = 1,
    [ValidateSet('','2k','32k')][string]$Only = '',
    [ValidateSet('rocm7','rocmfpx')][string]$Runtime = 'rocm7',
    [string]$OutFile = ''
)
$ErrorActionPreference = 'Continue'
$devRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if ($Runtime -eq 'rocmfpx') {
    $bin = "$PSScriptRoot\..\bin-rocmfpx\llama-cli.exe"
    # The fork's binaries are staged without the HIP runtime; they load it from the HIP SDK.
    if (-not $env:HIP_PATH) { throw "HIP_PATH is not set - the ROCmFPX build needs the HIP SDK at runtime." }
    $env:PATH = "$($env:HIP_PATH.TrimEnd('\'))\bin;$env:PATH"
} else {
    $bin = "$devRoot\llm-bench\bin\llama-cli.exe"
}
$model = "$env:USERPROFILE\.lmstudio\models\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q8_0.gguf"
$sp    = "$PSScriptRoot\..\data"
$out   = if ($OutFile) { $OutFile } else { "$PSScriptRoot\..\results\mtp-nmax-sweep.csv" }

if (-not (Test-Path $bin))   { throw "llama-cli.exe not found at $bin  (run ..\llm-bench\Setup.ps1)" }
if (-not (Test-Path $model)) { throw "Model not found: $model" }

# Derive the two prompt files from the existing 128K prose corpus (~4.37 chars/token here).
$src = "$sp\prompt-128k.txt"
if (-not (Test-Path $src)) { throw "Missing $src - build it with rocmfpx-fp4-mtp-128k.ps1 first." }
$corpus = [IO.File]::ReadAllText($src)
foreach ($p in @(@{f="$sp\prompt-2k.txt"; n=8500}, @{f="$sp\prompt-32k.txt"; n=140000})) {
    if (-not (Test-Path $p.f)) { [IO.File]::WriteAllText($p.f, $corpus.Substring(0, $p.n)) }
}

if (Test-Path $out) { Remove-Item $out }
'runtime,ctx_label,n_max,p_min,rep,prefill_tps,decode_tps' | Out-File $out -Encoding utf8

# This build does not emit the classic llama_perf timing block under --simple-io; it prints its own
# "[ Prompt: X t/s | Generation: Y t/s ]" summary instead (same source as the S5/S7 tables).
function ParseRun($log) {
    $line = Get-Content $log | Select-String '\[\s*Prompt:.*Generation:' | Select-Object -Last 1
    if ($line -and $line.Line -match 'Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s') {
        return @{ pp = [double]$Matches[1]; tg = [double]$Matches[2] }
    }
    return @{ pp = [double]::NaN; tg = [double]::NaN }
}

$cases = @(
    @{ label='2k';  file="$sp\prompt-2k.txt";  ctx=8192  },
    @{ label='32k'; file="$sp\prompt-32k.txt"; ctx=36000 }
)
# p_min '' = leave llama.cpp's default, which this build reports as 0.00 - i.e. the ciru profile's
# explicit "--spec-draft-p-min 0.0" is a no-op vs what we already run. The only real difference
# against their profile is the draft depth. LM Studio goes the other way (0.75), so test that too.
$configs = @(
    @{ nmax=4; pmin='' },    # our current default (Serve-Qwen.ps1)
    @{ nmax=6; pmin='' },    # ciru profile depth
    @{ nmax=8; pmin='' },    # does it keep scaling?
    @{ nmax=6; pmin='0.75' } # LM Studio-style conservative gate at the deeper draft
)

function RunOne($label, $file, $ctx, $nmax, $pmin, $rep) {
    $a = @('-m',$model,'-f',$file,'-n','256','-c',"$ctx",'-t','16','-ngl','-1','-fa','on',
           '-dev','ROCm0','--temp','0','--seed','123','--no-warmup','--simple-io',
           '--single-turn','--no-display-prompt','--keep','0',
           '--spec-type','draft-mtp','--spec-draft-n-max',"$nmax")
    if ($pmin -ne '') { $a += @('--spec-draft-p-min', $pmin) }

    $tag = "n$nmax$(if ($pmin -ne '') { "-pmin$pmin" })"
    Write-Host "`n======== $Runtime / $label / $tag / rep $rep ========" -ForegroundColor Yellow
    $pfx = if ($Runtime -eq 'rocmfpx') { 'mtp-nmax-fork' } else { 'mtp-nmax' }
    $log = "$PSScriptRoot\..\results\$pfx-$label-$tag$(if ($Repeat -gt 1) { "-r$rep" }).log"
    & $bin @a *>$log

    $r = ParseRun $log
    Write-Host ("  prefill t/s   : {0:N2}" -f $r.pp) -ForegroundColor White
    Write-Host ("  decode  t/s   : {0:N2}" -f $r.tg) -ForegroundColor Green

    "{0},{1},{2},{3},{4},{5:N2},{6:N2}" -f `
        $Runtime,$label,$nmax,$(if($pmin -ne ''){$pmin}else{'default (0.00)'}),$rep,$r.pp,$r.tg |
        Out-File $out -Append -Encoding utf8
}

foreach ($c in $cases) {
    if ($Only -and $c.label -ne $Only) { continue }
    # Interleave reps rather than looping a config N times back-to-back, so slow drift (thermals,
    # page cache) hits every config equally instead of penalising whichever ran last.
    for ($rep = 1; $rep -le $Repeat; $rep++) {
        foreach ($cfg in $configs) { RunOne $c.label $c.file $c.ctx $cfg.nmax $cfg.pmin $rep }
    }
}
Write-Host "`nDONE -> $out" -ForegroundColor Green
Get-Content $out

if ($Repeat -gt 1) {
    Write-Host "`nMedian decode t/s by config:" -ForegroundColor Cyan
    Import-Csv $out | Group-Object runtime,ctx_label,n_max,p_min | ForEach-Object {
        $tg = @($_.Group.decode_tps | ForEach-Object { [double]$_ } | Sort-Object)
        "{0,-28} median {1,5:N2}   min {2,5:N2}   max {3,5:N2}" -f `
            $_.Name, $tg[[int]($tg.Count/2)], $tg[0], $tg[-1]
    }
}
