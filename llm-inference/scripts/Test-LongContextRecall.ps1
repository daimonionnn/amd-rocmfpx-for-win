<#
.SYNOPSIS
  Needle-in-a-haystack recall at depth: does a 4-bit quant lose long-context recall against Q8?

.DESCRIPTION
  S13-S16 measured tool-calling quality at 32K and found Q4_K_M indistinguishable from Q8_0. The
  production workload runs at 128K+, and nothing in those evals touches long-context recall - the
  specific mechanism by which quantization damage would be expected to grow with depth. That gap
  is the only thing keeping Serve-Qwen.ps1 on Q8.

  Running the tool-call suite under context pressure costs ~3.6 h per arm and still only reaches
  49K. This measures the thing directly instead: bury facts at known depths in a large document
  and ask for them back.

  Affordable for the same reason context-pressure was: the haystack is a FIXED prefix, so the
  server prefills it once (~11 min at 100K) and every subsequent question matches it at high LCP
  similarity and costs only its own few hundred tokens. Change the haystack per question and this
  becomes a multi-hour job.

  Scoring is exact substring match on a distinctive token, so it is unambiguous - no judge, no
  partial credit, no room for interpretation.

.PARAMETER Model
  GGUF to test. Serve it yourself, or let -Serve start the fork binary.

.PARAMETER Depths
  Fractions of the haystack at which needles are planted.

.EXAMPLE
  .\Test-LongContextRecall.ps1 -Label Q8_0
  .\Test-LongContextRecall.ps1 -Label Q4_K_M -HaystackTokens 100000
#>
param(
    [string]$Label = 'model',
    [int]$Port = 8081,
    [int]$HaystackTokens = 100000,
    [double[]]$Depths = @(0.05,0.15,0.30,0.45,0.60,0.75,0.90,0.98),
    [string]$OutFile = ''
)
$ErrorActionPreference = 'Continue'

$sp   = "$PSScriptRoot\..\data"
$out  = if ($OutFile) { $OutFile } else { "$PSScriptRoot\..\results\longctx-recall.csv" }
$src  = "$sp\prompt-128k.txt"
if (-not (Test-Path $src)) { throw "Missing $src" }

# ~4.37 chars/token on this corpus, measured in the S12 sweep.
$corpus = [IO.File]::ReadAllText($src)
$chars  = [Math]::Min($corpus.Length, [int]($HaystackTokens * 4.37))
$hay    = $corpus.Substring(0, $chars)

# Distinctive, memorable, and impossible to answer from world knowledge or from the surrounding
# wikitext - so a correct answer can only come from retrieval, not from the prior.
# `expect` is a regex, deliberately: the first run scored a correct answer as a miss because the
# needle said "two" and the model replied "2". Accepting only one surface form measures formatting,
# not recall, and would inject noise straight into the quant comparison.
$needles = @(
    @{ key='ATLAS';   fact='The maintenance code for the Kestrel relay is ATLAS-7734.';       ask='What is the maintenance code for the Kestrel relay?';        expect='ATLAS-?7734' }
    @{ key='VERDIGRIS'; fact='Sector nine was repainted in verdigris on the third of March.';  ask='What colour was sector nine repainted, and on what date?';   expect='verdigris' }
    @{ key='BRAMBLE'; fact='Operator Bramble logged 412 hours before the transfer.';           ask='How many hours did operator Bramble log before the transfer?'; expect='412' }
    @{ key='OKAPI';   fact='The reserve cistern at Okapi Station holds 1,850 litres.';         ask='How many litres does the reserve cistern at Okapi Station hold?'; expect='1[,.]?850' }
    @{ key='QUILL';   fact='Shipment QUILL-22 was rerouted through Harbour Four.';             ask='Where was shipment QUILL-22 rerouted through?';              expect='Harbou?r (Four|4)' }
    @{ key='CINDER';  fact='The Cinder protocol requires two independent signatures.';         ask='How many independent signatures does the Cinder protocol require?'; expect='\b(two|2)\b' }
    @{ key='MARLOW';  fact='Marlow Bridge carries a load limit of 63 tonnes.';                 ask='What load limit does Marlow Bridge carry?';                  expect='63' }
    @{ key='FENWICK'; fact='The Fenwick archive was sealed in the year 1974.';                 ask='In what year was the Fenwick archive sealed?';               expect='1974' }
)

# Plant every needle in one haystack so the prefix stays identical across questions - that is what
# makes the prefix cache work and the whole test affordable.
$planted = $hay
for ($i = $Depths.Count - 1; $i -ge 0; $i--) {
    if ($i -ge $needles.Count) { continue }
    $pos = [int]($planted.Length * $Depths[$i])
    # Snap to a sentence boundary so the insertion does not split a word.
    $nl = $planted.IndexOf("`n", $pos)
    if ($nl -lt 0) { $nl = $pos }
    $planted = $planted.Substring(0, $nl) + "`n`n" + $needles[$i].fact + "`n`n" + $planted.Substring($nl)
}

if (-not (Test-Path $out)) { 'label,needle,depth,expected,found,finish_reason,answer' | Out-File $out -Encoding utf8 }

Write-Host "`n=== $Label : haystack ~$HaystackTokens tokens, $($needles.Count) needles ===" -ForegroundColor Cyan
Write-Host "First question pays the full prefill; the rest reuse it." -ForegroundColor DarkGray

$hits = 0; $n = 0
for ($i = 0; $i -lt $needles.Count -and $i -lt $Depths.Count; $i++) {
    $q = $needles[$i]
    # This model is served with reasoning enabled, so it emits a <think> block before answering and
    # the visible content stays empty until that finishes. A small max_tokens truncates inside the
    # reasoning and returns an empty answer for every question - which reads as total recall
    # failure and is nothing of the sort. Leave enough room to think AND answer, and check
    # finish_reason so a truncation can never be mistaken for a miss again.
    $body = @{
        messages = @(
            @{ role='user'; content = $planted + "`n`n---`n" + $q.ask + " Answer with only the value, no explanation." }
        )
        max_tokens = 800
        temperature = 0
        seed = 42
    } | ConvertTo-Json -Depth 5 -Compress

    $t0 = Get-Date; $fin = ''
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
             -ContentType 'application/json' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 1800
        $msg = $r.choices[0].message
        $fin = $r.choices[0].finish_reason
        $ans = ($msg.content -replace '\s+',' ').Trim()
        # Fall back to the reasoning text only when the answer field is empty: the value being
        # present there still demonstrates retrieval, and it is flagged separately below.
        if (-not $ans -and $msg.reasoning_content) { $ans = '[reasoning] ' + (($msg.reasoning_content -replace '\s+',' ').Trim()) }
    } catch { $ans = "ERROR: $($_.Exception.Message)" }
    $secs = [math]::Round(((Get-Date) - $t0).TotalSeconds)
    if ($fin -eq 'length') { Write-Host "  (truncated - raise max_tokens)" -ForegroundColor Yellow }

    $found = $ans -match $q.expect
    if ($found) { $hits++ }
    $n++
    "{0,-10} depth {1,5:P0}  {2,-12} {3,-5} {4,4}s  {5}" -f `
        $q.key, $Depths[$i], $q.expect, $(if ($found) {'HIT'} else {'MISS'}), $secs, $ans.Substring(0,[Math]::Min(60,$ans.Length))

    '"{0}","{1}",{2},"{3}",{4},"{5}","{6}"' -f $Label,$q.key,$Depths[$i],$q.expect,$found,$fin,($ans -replace '"',"'") |
        Out-File $out -Append -Encoding utf8
}

Write-Host ("`n{0}: {1}/{2} recalled = {3:P0}" -f $Label, $hits, $n, ($hits/$n)) -ForegroundColor Green
