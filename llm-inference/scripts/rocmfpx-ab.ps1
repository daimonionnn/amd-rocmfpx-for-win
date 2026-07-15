# A/B: the ROCmFPX fork + its ROCmFP4 weight format vs our production ROCm 7 build + Q8_0.
#
# Four configs, chosen so the format effect and the runtime effect can be told apart:
#   1. q8-rocm7      Q8_0    on bin\          ROCm0   <- current production default (baseline)
#   2. q8-fpx        Q8_0    on bin-rocmfpx\  ROCm0   <- CONTROL: same weights, other runtime.
#                                                        Isolates "fork build" from "FP4 format".
#   3. fp4-fpx       ROCmFP4 on bin-rocmfpx\  ROCm0   <- the actual proposition
#   4. fp4-fpx-vk    ROCmFP4 on bin-rocmfpx\  Vulkan0 <- upstream calls Vulkan its stronger
#                                                        decode path on Strix Halo; test it.
#
# llama-bench has no MTP/speculative support, so these are raw prefill/decode numbers -
# directly comparable to results\decode-quant-compare.csv, NOT to the +MTP numbers in README §5.
#
# -Full adds the 128K points. Budget for it: a single 128K prefill is ~15 min at ~150 t/s, and
# llama-bench redoes that prefill for EVERY rep and again for the -d 131072 decode point. So the
# long points run at -r 1 (the short ones stay at -r 2) - otherwise the sweep is a 5-hour job.
# Even so, expect ~30 min per config, ~2h for all four.
#
# -Tag labels the output file (results\rocmfpx-ab[-tag].csv) so a re-run under different machine
# state does not clobber the previous one. Used for the BIOS IOMMU on/off comparison:
#   .\rocmfpx-ab.ps1 -Tag iommu-off
param([switch]$Full, [string]$Tag = '')

$ErrorActionPreference = 'Continue'
$benchRocm7 = 'c:\development\ai-tools-for-win\llm-bench\bin\llama-bench.exe'
$benchFpx   = 'c:\development\ai-tools-for-win\llm-inference\bin-rocmfpx\llama-bench.exe'
$q8         = "$env:USERPROFILE\.lmstudio\models\unsloth\Qwen3.6-27B-MTP-GGUF\Qwen3.6-27B-Q8_0.gguf"
$fp4        = 'c:\development\ai-tools-for-win\llm-inference\models\Qwen3.6-27B-MTP-ROCmFP4-STRIX-imatrix-embF16-headQ6.gguf'
$suffix     = if ($Tag) { "-$Tag" } else { '' }
$out        = "$PSScriptRoot\..\results\rocmfpx-ab$suffix.csv"

# The fork's binaries are staged without the HIP runtime - it lives in the HIP SDK.
if (-not $env:HIP_PATH) { throw "HIP_PATH is not set (HIP SDK required by bin-rocmfpx\)." }
$env:PATH = "$($env:HIP_PATH.TrimEnd('\'))\bin;$env:PATH"

$configs = [ordered]@{
    'q8-rocm4'   = @{ Bin = $benchRocm7; Model = $q8;  Dev = 'ROCm0'   }
    'q8-fpx'     = @{ Bin = $benchFpx;   Model = $q8;  Dev = 'ROCm0'   }
    'fp4-fpx'    = @{ Bin = $benchFpx;   Model = $fp4; Dev = 'ROCm0'   }
    'fp4-fpx-vk' = @{ Bin = $benchFpx;   Model = $fp4; Dev = 'Vulkan0' }
}
# (key 'q8-rocm4' is the ROCm 7 baseline; name kept short for the CSV column)

if (Test-Path $out) { Remove-Item $out }

foreach ($name in $configs.Keys) {
    $c = $configs[$name]
    if (-not (Test-Path $c.Bin))   { Write-Host "SKIP $name (no binary: $($c.Bin))"   -ForegroundColor Red; continue }
    if (-not (Test-Path $c.Model)) { Write-Host "SKIP $name (no model: $($c.Model))"  -ForegroundColor Red; continue }

    Write-Host "`n============ $name  ($($c.Dev), $([IO.Path]::GetFileName($c.Model))) ============" -ForegroundColor Yellow

    # Prefill-only (-n 0) and decode-only (-p 0) at depth, same as the other scripts.
    # The 128K points are split into their own -r 1 passes (see the header comment on cost).
    # 3 reps on the short passes: pp32768 on this box has ~±14% run-to-run spread (thermal/clock),
    # enough that 2 reps produced a physically impossible curve (pp32768 > pp16384) once.
    $passes = @(
        @{ Tag = 'prefill'; Reps = 3; Args = @('-p','4096,16384,32768','-n','0','-ub','1024') }
        @{ Tag = 'decode';  Reps = 3; Args = @('-p','0','-n','128','-d','0,32768') }
    )
    if ($Full) {
        $passes += @(
            @{ Tag = 'prefill'; Reps = 1; Args = @('-p','131072','-n','0','-ub','1024') }
            @{ Tag = 'decode';  Reps = 1; Args = @('-p','0','-n','128','-d','131072') }
        )
    }
    foreach ($p in $passes) {
        $args = @('-m',$c.Model,'-dev',$c.Dev,'-ngl','-1','-fa','on','-t','16','-r',"$($p.Reps)") + $p.Args
        Write-Host "> llama-bench $($args -join ' ')" -ForegroundColor DarkGray

        # ONE run per pass, in CSV form, and the console table is rendered from that same data.
        # (The sibling scripts run llama-bench twice - once -o md to show, once -o csv to log.
        # That doubles an already multi-minute sweep AND logs different numbers than it prints:
        # a 32K prefill measured 200 t/s in one run and 289 t/s in the other. Never do that here.)
        $csv  = & $c.Bin @args -o csv 2>$null
        $code = $LASTEXITCODE
        if ($code -ne 0) { Write-Host "  ($name/$($p.Tag) FAILED, code $code)" -ForegroundColor Red; continue }

        $lines = $csv -split "`r?`n" | Where-Object { $_ -ne '' }
        if (-not (Test-Path $out)) { "config,pass,$($lines[0])" | Out-File $out -Encoding utf8 }
        foreach ($d in ($lines | Select-Object -Skip 1)) { "$name,$($p.Tag),$d" | Out-File $out -Append -Encoding utf8 }

        # Show what was just logged: test name (pp4096 / tg128 @ d32768 ...) + t/s.
        $lines | ConvertFrom-Csv | ForEach-Object {
            $test = if ([int]$_.n_prompt -gt 0) { "pp$($_.n_prompt)" }
                    elseif ([int]$_.n_depth -gt 0) { "tg$($_.n_gen) @ d$($_.n_depth)" }
                    else { "tg$($_.n_gen)" }
            Write-Host ("    {0,-18} {1,8:N2} t/s" -f $test, [double]$_.avg_ts)
        }
    }
}
Write-Host "`nDONE -> $out" -ForegroundColor Green
