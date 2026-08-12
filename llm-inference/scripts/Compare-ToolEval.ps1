<#
.SYNOPSIS
  Diff two tool-eval-bench JSON runs: overall, per category, and per scenario.

.DESCRIPTION
  The headline score hides what matters in an A/B. Two runs can tie at 83 with completely
  different failure sets - scattered single-point drops are noise, drops concentrated in one or
  two categories are real degradation. This prints both views and lists every scenario that
  changed, so the shape of the difference is visible rather than just its magnitude.

  Also surfaces the safety gate independently: it is a pass/fail gate, not part of the score, so
  a run can lose the gate while its number goes up.

.PARAMETER Baseline
  JSON produced by `tool-eval-bench --json-file` for the reference config.

.PARAMETER Candidate
  JSON for the config under test.

.EXAMPLE
  .\Compare-ToolEval.ps1 -Baseline q8-full.json -Candidate fp4-full.json
#>
param(
    [Parameter(Mandatory)][string]$Baseline,
    [Parameter(Mandatory)][string]$Candidate
)
$ErrorActionPreference = 'Stop'

$a = Get-Content $Baseline  -Raw | ConvertFrom-Json
$b = Get-Content $Candidate -Raw | ConvertFrom-Json

function Label($j, $path) {
    if ($j.metadata.label) { $j.metadata.label } else { [IO.Path]::GetFileNameWithoutExtension($path) }
}
$la = Label $a $Baseline
$lb = Label $b $Candidate

# Guard the comparison itself: a differing seed, scenario count or temperature makes the diff
# meaningless, and that is easy to do by accident across two long runs.
foreach ($f in 'seed','scenario_count','temperature') {
    if ($a.config.$f -ne $b.config.$f) {
        Write-Host "WARNING: config.$f differs ($($a.config.$f) vs $($b.config.$f)) - runs are not comparable" -ForegroundColor Red
    }
}
# --scenarios silently drops hard-mode IDs unless --hardmode is also passed (it gates what is
# selectable, it does not just append). A run can therefore come back short with no error at all.
if ($a.scores.scenario_results.Count -ne $b.scores.scenario_results.Count) {
    Write-Host "WARNING: scenario counts differ ($($a.scores.scenario_results.Count) vs $($b.scores.scenario_results.Count)) - some were dropped from a selection" -ForegroundColor Red
}
# max_points, not scenario count, is the scoring denominator: infrastructure failures (timeouts,
# connection errors) stay in the results list but are removed from max, so two runs over the same
# 84 scenarios can still be scored out of different totals. Compare on a common basis.
if ($a.scores.max_points -ne $b.scores.max_points) {
    Write-Host ("NOTE: different scoring denominators ({0} vs {1} max points) - an infrastructure" -f $a.scores.max_points, $b.scores.max_points) -ForegroundColor Yellow
    Write-Host ("      failure was excluded from one run. On a common {0}-point basis: {1} = {2:N1}, {3} = {4:N1}" -f `
        [math]::Max($a.scores.max_points,$b.scores.max_points), `
        $la, (100*$a.scores.total_points/[math]::Max($a.scores.max_points,$b.scores.max_points)), `
        $lb, (100*$b.scores.total_points/[math]::Max($a.scores.max_points,$b.scores.max_points))) -ForegroundColor Yellow
}

Write-Host "`n=== Overall ===" -ForegroundColor Cyan
[pscustomobject]@{
    Metric = 'final_score'; $la = $a.final_score; $lb = $b.final_score
    Delta  = '{0:+#;-#;0}' -f ($b.final_score - $a.final_score)
} | Format-Table -AutoSize
"{0,-22} {1,10} {2,10}" -f 'points', "$($a.scores.total_points)/$($a.scores.max_points)", "$($b.scores.total_points)/$($b.scores.max_points)"
"{0,-22} {1,10} {2,10}" -f 'rating',       $a.rating,        $b.rating
"{0,-22} {1,10} {2,10}" -f 'safety_gate',  $a.safety_gate.passed, $b.safety_gate.passed
"{0,-22} {1,10} {2,10}" -f 'deployability',$a.deployability, $b.deployability
"{0,-22} {1,10} {2,10}" -f 'responsiveness',$a.responsiveness,$b.responsiveness
"{0,-22} {1,10} {2,10}" -f 'median_turn_ms',[math]::Round($a.scores.median_turn_ms), [math]::Round($b.scores.median_turn_ms)
"{0,-22} {1,10} {2,10}" -f 'total_tokens', $a.scores.total_tokens, $b.scores.total_tokens

Write-Host "`n=== Per category ===" -ForegroundColor Cyan
$catB = @{}; foreach ($c in $b.scores.category_scores) { $catB[$c.category] = $c }
$a.scores.category_scores | ForEach-Object {
    $o = $catB[$_.category]
    [pscustomobject]@{
        Cat = $_.category; Label = $_.label
        $la = "$($_.percent)%"; $lb = "$($o.percent)%"
        Delta = '{0:+#;-#;0}' -f ($o.percent - $_.percent)
    }
} | Sort-Object { [int]($_.Delta) } | Format-Table -AutoSize

Write-Host "`n=== Scenarios that changed ===" -ForegroundColor Cyan
# No category per scenario here: scenario_results has no category field and category_scores only
# carries aggregates, so the id -> category mapping exists solely in the streamed NDJSON events.
# Read the per-category table above for where the change landed.
# Excluded scenarios are the trap here. An infrastructure failure (timeout, connection error) is
# still listed with status "fail" and 0 points, but removed from max_points - so it looks like a
# content failure and is not one. A run that got slower for unrelated reasons will sprout several
# of these and read as a quality regression. They have no note and no summary; flag them instead
# of scoring them.
function IsExcluded($r) { -not $r.note -and -not $r.summary -and $r.status -eq 'fail' }
$exA = @($a.scores.scenario_results | Where-Object { IsExcluded $_ })
$exB = @($b.scores.scenario_results | Where-Object { IsExcluded $_ })
if ($exA.Count -or $exB.Count) {
    Write-Host ("NOTE: infrastructure failures excluded from scoring - {0}: {1}   {2}: {3}" -f `
        $la, $(if ($exA) { ($exA.scenario_id) -join ',' } else { 'none' }),
        $lb, $(if ($exB) { ($exB.scenario_id) -join ',' } else { 'none' })) -ForegroundColor Yellow
    Write-Host "      These report status=fail but did not fail on content. Do not read them as regressions." -ForegroundColor Yellow
}

$resB = @{}; foreach ($r in $b.scores.scenario_results) { $resB[$r.scenario_id] = $r }
$changed = $a.scores.scenario_results | ForEach-Object {
    $o = $resB[$_.scenario_id]
    if ($o -and $o.points -ne $_.points) {
        [pscustomobject]@{
            Scenario = $_.scenario_id
            $la = $_.status; $lb = $o.status
            Delta = '{0:+#;-#;0}' -f ($o.points - $_.points)
            Excluded = $(if ((IsExcluded $_) -or (IsExcluded $o)) { 'TIMEOUT' } else { '' })
        }
    }
}
if ($changed) { $changed | Sort-Object { [int]($_.Delta) } | Format-Table -AutoSize }
else { Write-Host "  (none - identical scenario-level outcomes)" -ForegroundColor DarkGray }

$regressions = @($changed | Where-Object { [int]$_.Delta -lt 0 }).Count
$gains       = @($changed | Where-Object { [int]$_.Delta -gt 0 }).Count
Write-Host "`n$lb vs ${la}: $regressions scenarios worse, $gains better, $(84 - @($changed).Count) unchanged" -ForegroundColor White
Write-Host "Interpretation: scattered +/-1 swaps are run-to-run noise; several regressions inside one" -ForegroundColor DarkGray
Write-Host "category are real. Check the safety gate separately - it is not part of the score." -ForegroundColor DarkGray
