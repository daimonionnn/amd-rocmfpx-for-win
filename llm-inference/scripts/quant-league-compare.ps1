<#
.SYNOPSIS
  Same-size quant comparison: ROCmFPX formats against conventional quants of equal weight.

.DESCRIPTION
  Earlier work compared ROCmFP4 (15.7 GiB) against Q8_0 (27.0 GiB) and concluded 4-bit costs
  quality. True, but trivial - it compares leagues. The question that actually tests ROCmFPX's
  claim is whether its formats beat a CONVENTIONAL quant of the same size:

      4-bit league   ROCmFP4        15.70 GiB   vs  Q4_K_M 15.40, UD-Q4_K_XL 16.67
      6-bit league   Chadrockv2 FP6 23.47 GiB   vs  Q6_K   20.60, UD-Q6_K_XL 24.20

  Two peers per league on purpose, because they separate two different questions. The plain K-quant
  is the closest size match, so it isolates the FORMAT. The UD "XL" variant protects token
  embeddings and the output head - structurally the same idea as ROCmFP4's own embF16/headQ6 - so
  it isolates the PROTECTION RECIPE. (Only the XL variants are Unsloth Dynamic; Q4_K_M and Q6_K are
  conventional.)

  The bandwidth rule (S4: t/s x GiB ~ 198) predicts each peer's decode purely from its size.
  Beating exactly that margin means the format adds nothing; beating it by more is a real kernel
  edge. That is the measurement.

  Every model runs on the SAME runtime - the fork, since the ROCmFPX files require it. S10 measured
  that swapping runtimes moves results as much as 4-bit quantization does, so runtime has to be
  held constant or the comparison is meaningless. This is the same trap that made an earlier FP4
  result read as a tie.

  Depths cover what the agent actually sees, not just the 128K worst case: plenty of requests land
  at 16K-64K, where a smaller model's decode advantage is largest and most relevant.

.PARAMETER Depths
  Prompt sizes to test. Default 0k,2k,16k,32k,64k. Add 128k for the deep point (~16 min per model).
  0k is a ~50-token prompt: decode there is the pure weight-streaming number with no meaningful KV
  traffic, which is the cleanest test of the bandwidth rule. Its prefill figure is dominated by
  fixed per-request overhead and should not be read as throughput.

.PARAMETER Models
  Restrict to a subset by key (fp4, q4km, q4xl, fp6, q6k, q6xl, q8).

.EXAMPLE
  .\quant-league-compare.ps1
  .\quant-league-compare.ps1 -Models fp4,q4km,q4xl -Depths 0k,32k
#>
param(
    [string[]]$Depths = @('0k','2k','16k','32k','64k'),
    [string[]]$Models = @(),
    [int]$DraftNMax = 6,
    # MTP multiplies decode by a draft-acceptance factor that differs per model, so a served
    # comparison measures bandwidth AND drafting quality together. That matters here: ROCmFP4 keeps
    # its output head at Q6 (embF16-headQ6) while its body is 4-bit, which should draft better than
    # a uniformly 4-bit peer. Run with -NoMtp to get raw decode and separate the two.
    [switch]$NoMtp,
    [string]$OutFile = ''
)
$ErrorActionPreference = 'Continue'

$lm  = "$env:USERPROFILE\.lmstudio\models"
$bin = "$PSScriptRoot\..\bin-rocmfpx\llama-cli.exe"
$sp  = "$PSScriptRoot\..\data"
$out = if ($OutFile) { $OutFile } else { "$PSScriptRoot\..\results\quant-league.csv" }

if (-not $env:HIP_PATH) { throw "HIP_PATH is not set - the ROCmFPX build needs the HIP SDK at runtime." }
$env:PATH = "$($env:HIP_PATH.TrimEnd('\'))\bin;$env:PATH"
if (-not (Test-Path $bin)) { throw "Fork llama-cli not found at $bin (run .\Setup-ROCmFPX.ps1)" }

# Sizes are read from disk below rather than hard-coded: every conclusion here rests on
# decode x GiB and on how much of a lead a model's size alone predicts, so a stale figure would
# quietly corrupt the verdict.
$catalog = @(
    @{ key='fp4';  league='4-bit'; gib=15.70; label='ROCmFP4'; fpx=$true;
       path="$PSScriptRoot\..\models\Qwen3.6-27B-MTP-ROCmFP4-STRIX-imatrix-embF16-headQ6.gguf" },
    @{ key='q4km'; league='4-bit'; gib=15.40; label='Q4_K_M'; fpx=$false;
       path="$lm\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q4_K_M.gguf" },
    @{ key='q4xl'; league='4-bit'; gib=16.67; label='UD-Q4_K_XL'; fpx=$false;
       path="$lm\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-UD-Q4_K_XL.gguf" },
    @{ key='fp6';  league='6-bit'; gib=23.47; label='Chadrockv2-FP6'; fpx=$true;
       path="$lm\jcbtc\Chadrockv2-Qwen3.6-27B-ROCmFP6-STRIX-QUALITY\Chadrockv2-Qwen3.6-27B-ROCmFP6-STRIX-QUALITY.gguf" },
    @{ key='q6k';  league='6-bit'; gib=20.60; label='Q6_K'; fpx=$false;
       path="$lm\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q6_K.gguf" },
    @{ key='q6xl'; league='6-bit'; gib=24.20; label='UD-Q6_K_XL'; fpx=$false;
       path="$lm\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-UD-Q6_K_XL.gguf" },
    @{ key='q8';   league='ref';   gib=27.05; label='Q8_0'; fpx=$false;
       path="$lm\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q8_0.gguf" },
    # The actual ceiling. Everything else in this repo treats Q8 as "reference", which is a
    # borrowed assumption (Q8 is widely held to be near-lossless) rather than something measured
    # here. Point llama.cpp at shard 1; it picks up the rest.
    # Restricted to shallow depths on purpose: 50.9 GiB of weights plus a 64K KV cache exceeds the
    # 64 GB VRAM carve-out, so deeper rows would measure paging, not the model.
    @{ key='bf16'; league='ref';   gib=50.90; label='BF16'; fpx=$false; only=@('0k','2k','16k');
       path="$lm\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-BF16-00001-of-00002.gguf" }
)
if ($Models.Count) { $catalog = @($catalog | Where-Object { $Models -contains $_.key }) }

$missing = @($catalog | Where-Object { -not (Test-Path $_.path) })
if ($missing) {
    Write-Host "Missing model files - skipping:" -ForegroundColor Yellow
    $missing | ForEach-Object { Write-Host "  $($_.label): $($_.path)" -ForegroundColor DarkGray }
    $catalog = @($catalog | Where-Object { Test-Path $_.path })
}
if (-not $catalog) { throw "No model files available." }

# Measure sizes now. Split GGUFs (BF16) are summed across shards - llama.cpp streams all of them,
# so the weight traffic that the bandwidth rule cares about is the total, not shard 1.
foreach ($m in $catalog) {
    $files = if ($m.path -match '-(\d{5})-of-(\d{5})\.gguf$') {
        Get-ChildItem (Split-Path $m.path) -Filter ($m.path -replace '.*\\','' -replace '-\d{5}-of-\d{5}\.gguf$','-*-of-*.gguf')
    } else { Get-Item $m.path }
    $m.gib = [math]::Round((($files | Measure-Object Length -Sum).Sum) / 1GB, 2)
}
Write-Host "Models under test:" -ForegroundColor Cyan
$catalog | ForEach-Object { "  {0,-16} {1,6:N2} GiB  {2}" -f $_.label, $_.gib, $_.league }

# Prompt files, derived from the 128K prose corpus at ~4.37 chars/token.
$src = "$sp\prompt-128k.txt"
if (-not (Test-Path $src)) { throw "Missing $src - build it with rocmfpx-fp4-mtp-128k.ps1 first." }
$corpus = [IO.File]::ReadAllText($src)
$depthSpec = @{
    '0k'   = @{ chars=220;    ctx=4096   }   # ~50 tokens: pure weight-streaming decode
    '2k'   = @{ chars=8500;   ctx=8192   }
    '16k'  = @{ chars=70000;  ctx=20000  }
    '32k'  = @{ chars=140000; ctx=36000  }
    '64k'  = @{ chars=280000; ctx=70000  }
    '128k' = @{ chars=600000; ctx=138000 }
}
foreach ($d in $Depths) {
    $f = "$sp\prompt-$d.txt"
    if (-not (Test-Path $f)) { [IO.File]::WriteAllText($f, $corpus.Substring(0, $depthSpec[$d].chars)) }
}

if (Test-Path $out) { Remove-Item $out }
'league,model,gib,depth,prefill_tps,decode_tps,decode_x_gib' | Out-File $out -Encoding utf8

# This build prints its own summary instead of the classic llama_perf block under --simple-io.
function ParseRun($log) {
    $line = Get-Content $log | Select-String '\[\s*Prompt:.*Generation:' | Select-Object -Last 1
    if ($line -and $line.Line -match 'Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s') {
        return @{ pp = [double]$Matches[1]; tg = [double]$Matches[2] }
    }
    return @{ pp = [double]::NaN; tg = [double]::NaN }
}

foreach ($m in $catalog) {
    foreach ($d in $Depths) {
        # Entries may cap their own depth range when the weights leave no room for the KV cache.
        if ($m.only -and $m.only -notcontains $d) {
            Write-Host "`n-------- $($m.label) / $d : skipped (would not fit alongside the KV cache)" -ForegroundColor DarkGray
            continue
        }
        $spec = $depthSpec[$d]
        Write-Host "`n======== $($m.label) / $d ========" -ForegroundColor Yellow
        $log = "$PSScriptRoot\..\results\league-$($m.key)-$d$(if ($NoMtp) { '-nomtp' }).log"
        $a = @('-m',$m.path,'-f',"$sp\prompt-$d.txt",'-n','256','-c',"$($spec.ctx)",'-t','16',
               '-ngl','-1','-fa','on','-dev','ROCm0','--temp','0','--seed','123','--no-warmup',
               '--simple-io','--single-turn','--no-display-prompt','--keep','0')
        if (-not $NoMtp) { $a += @('--spec-type','draft-mtp','--spec-draft-n-max',"$DraftNMax") }
        & $bin @a *>$log

        $r = ParseRun $log
        # decode x GiB is the bandwidth-rule product: flat across quants means the format adds
        # nothing beyond being smaller; a higher product is a genuine kernel advantage.
        $prod = if ([double]::IsNaN($r.tg)) { [double]::NaN } else { $r.tg * $m.gib }
        Write-Host ("  prefill {0,7:N1} t/s   decode {1,6:N2} t/s   decode x GiB {2,6:N0}" -f $r.pp, $r.tg, $prod)

        "{0},{1},{2},{3},{4:N2},{5:N2},{6:N0}" -f $m.league,$m.label,$m.gib,$d,$r.pp,$r.tg,$prod |
            Out-File $out -Append -Encoding utf8
    }
}

Write-Host "`nDONE -> $out" -ForegroundColor Green
Import-Csv $out | Format-Table -AutoSize

Write-Host "`nLeague deltas - ROCmFPX vs each conventional peer, same runtime, same prompt:" -ForegroundColor Cyan
Write-Host "  'real edge' = measured decode advantage minus the advantage its smaller size alone predicts." -ForegroundColor DarkGray
$fpxKeys = @{ '4-bit' = 'ROCmFP4'; '6-bit' = 'Chadrockv2-FP6' }
$all = @(Import-Csv $out)
foreach ($lg in @('4-bit','6-bit')) {
    $rows = @($all | Where-Object league -eq $lg)
    if (-not $rows) { continue }
    foreach ($d in ($rows.depth | Select-Object -Unique)) {
        $atDepth = @($rows | Where-Object depth -eq $d)
        $fpx = $atDepth | Where-Object model -eq $fpxKeys[$lg]
        if (-not $fpx -or [double]::IsNaN([double]$fpx.decode_tps)) { continue }
        foreach ($peer in @($atDepth | Where-Object model -ne $fpxKeys[$lg])) {
            if ([double]::IsNaN([double]$peer.decode_tps)) { continue }
            $sizeEdge = 100 * ([double]$peer.gib / [double]$fpx.gib - 1)
            $realEdge = 100 * ([double]$fpx.decode_tps / [double]$peer.decode_tps - 1)
            $gap = $realEdge - $sizeEdge
            $verdict = if ($gap -gt 3) { 'real kernel edge' }
                       elseif ($gap -lt -3) { 'WORSE than size implies' }
                       else { 'size only, no format gain' }
            "{0,-6} {1,-5} vs {2,-12} measured {3,6:N1}%  size predicts {4,5:N1}%  gap {5,5:N1}%  -> {6}" -f `
                $lg, $d, $peer.model, $realEdge, $sizeEdge, $gap, $verdict
        }
    }
}
