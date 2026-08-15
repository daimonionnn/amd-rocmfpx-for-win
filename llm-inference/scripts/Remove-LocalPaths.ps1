<#
.SYNOPSIS
  Replace absolute user paths in tracked result files with %USERPROFILE%.

.DESCRIPTION
  The scripts in this repo are portable - every one resolves models through $env:USERPROFILE - but
  the tools they drive are not. llama-bench records its full command line into each CSV row, and
  tool-eval-bench stores the model path in run metadata, so committed measurement data ends up
  carrying the machine's own user path.

  That is not a secret, but it is noise in a public repo and it makes results look
  machine-specific when they are not. Run this before committing new results.

  Handles all three escapings that occur in practice: plain Windows (CSV, logs), forward-slash
  (some tool output), and JSON-escaped double backslashes (eval JSON).

.PARAMETER User
  Username to strip. Defaults to whoever is running it, which is the case that matters.

.PARAMETER WhatIf
  Report what would change without writing.

.EXAMPLE
  .\Remove-LocalPaths.ps1
  .\Remove-LocalPaths.ps1 -WhatIf
#>
param(
    [string]$User = $env:USERNAME,
    [switch]$WhatIf
)
$ErrorActionPreference = 'Stop'

$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Push-Location $repo

$bs = [char]92
$subs = @(
    @{ from = "C:$bs$bs" + "Users$bs$bs$User$bs$bs"; to = "%USERPROFILE%$bs$bs" }  # JSON-escaped
    @{ from = "C:$bs" + "Users$bs$User$bs";          to = "%USERPROFILE%$bs"      }  # plain
    @{ from = "C:/Users/$User/";                     to = '%USERPROFILE%/'        }  # forward slash
)

$changed = 0
foreach ($f in (git ls-files)) {
    if (-not (Test-Path $f -PathType Leaf)) { continue }
    try { $t = [IO.File]::ReadAllText($f) } catch { continue }
    $orig = $t
    foreach ($s in $subs) { $t = $t.Replace($s.from, $s.to) }
    if ($t -ne $orig) {
        $changed++
        if ($WhatIf) { Write-Host "  would rewrite $f" -ForegroundColor Yellow }
        else { [IO.File]::WriteAllText($f, $t); Write-Host "  rewrote $f" -ForegroundColor DarkGray }
    }
}

Pop-Location
if ($changed -eq 0) { Write-Host "Clean - no absolute user paths in tracked files." -ForegroundColor Green }
else { Write-Host "$changed file(s)$(if ($WhatIf) { ' would be' } else { '' }) rewritten." -ForegroundColor Green }
