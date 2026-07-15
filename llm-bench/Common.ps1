# Common.ps1 - shared config & helpers for the Strix Halo llama.cpp benchmark harness
# Dot-source this from the test scripts:  . "$PSScriptRoot\Common.ps1"

$ErrorActionPreference = 'Stop'

# --- Paths -------------------------------------------------------------------
$Root      = $PSScriptRoot
$BinDir    = Join-Path $Root 'bin'
$ResultsDir= Join-Path $Root 'results'
$Bench     = Join-Path $BinDir 'llama-bench.exe'
$Cli       = Join-Path $BinDir 'llama-cli.exe'

$LmModels  = Join-Path $env:USERPROFILE '.lmstudio\models'
$GemmaModel = Join-Path $LmModels 'lmstudio-community\gemma-3-27B-it-qat-GGUF\gemma-3-27B-it-QAT-Q4_0.gguf'
# MoE (35B total / ~3B active per token) - NOT a pure bandwidth proxy like a dense model.
$QwenMoeModel = Join-Path $LmModels 'lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf'
# Qwen MTP model folder from LM Studio. If this is a directory, first *.gguf is used.
$QwenMtpModel = Join-Path $LmModels 'unsloth\Qwen3.6-27B-MTP-GGUF'

# --- Helpers -----------------------------------------------------------------
function Assert-Bench {
    if (-not (Test-Path $Bench)) {
        throw "llama-bench.exe not found at $Bench`nRun .\Setup.ps1 first."
    }
}

function Assert-Cli {
    if (-not (Test-Path $Cli)) {
        throw "llama-cli.exe not found at $Cli`nRun .\Setup.ps1 first."
    }
}

function Assert-Model([string]$path, [string]$name) {
    if (-not (Test-Path $path)) {
        throw "$name model not found at:`n  $path`nDownload it in LM Studio first."
    }
}

function Resolve-ModelPath([string]$PathLike, [string]$ModelName) {
    if (-not (Test-Path $PathLike)) {
        throw "$ModelName model not found at:`n  $PathLike`nDownload it in LM Studio first."
    }

    $item = Get-Item -LiteralPath $PathLike
    if ($item.PSIsContainer) {
        $gguf = Get-ChildItem -LiteralPath $item.FullName -Filter '*.gguf' -File |
            Where-Object { $_.BaseName -notmatch '^(mmproj|mmproj-)' } |
            Sort-Object @{ Expression = {
                    if ($_.Name -match '00001-of-') { return 0 }
                    if ($_.Name -match 'Q4|IQ4|Q5|IQ5') { return 1 }
                    if ($_.Name -match 'Q6|IQ6') { return 2 }
                    if ($_.Name -match 'Q8|BF16|F16') { return 3 }
                    return 4
                }
            }, Name |
            Select-Object -First 1
        if (-not $gguf) {
            throw "$ModelName folder has no .gguf file:`n  $PathLike"
        }
        return $gguf.FullName
    }

    return $item.FullName
}

# Detects the current physical RAM (total GB + per-module capacities) so results
# are self-labelling (e.g. rocm-..._128GB.csv on this box).
function Get-RamTag {
    $mods = Get-CimInstance Win32_PhysicalMemory
    $totalGB = [math]::Round(($mods | Measure-Object -Property Capacity -Sum).Sum / 1GB)
    $sticks  = ($mods | ForEach-Object { [math]::Round($_.Capacity/1GB) }) -join '+'
    $speed   = ($mods | Select-Object -First 1 -ExpandProperty ConfiguredClockSpeed)
    [pscustomobject]@{
        TotalGB = $totalGB
        Sticks  = $sticks
        Count   = $mods.Count
        SpeedMTs= $speed
        Tag     = "${totalGB}GB"
    }
}

function Get-CpuInfo {
    $c = Get-CimInstance Win32_Processor | Select-Object -First 1
    [pscustomobject]@{
        Name    = $c.Name.Trim()
        Cores   = $c.NumberOfCores
        Threads = $c.NumberOfLogicalProcessors
    }
}

function Resolve-LlamaDevice([string]$BinaryPath, [string]$RequestedDevice) {
    if ($RequestedDevice -and $RequestedDevice -ne 'auto') {
        return $RequestedDevice
    }

    $eap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $BinaryPath --list-devices 2>&1
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $eap }

    if ($code -ne 0) {
        return $RequestedDevice
    }

    $deviceLine = ($out | Where-Object { $_ -match '^\s*[A-Za-z0-9_-]+:' } | Select-Object -First 1)
    if (-not $deviceLine) {
        return $RequestedDevice
    }

    if ($deviceLine -match '^\s*([A-Za-z0-9_-]+):') {
        return $Matches[1]
    }

    return $RequestedDevice
}

# Runs llama-bench, echoes the human-readable markdown table to the console,
# AND appends a CSV row set to a per-test results file tagged with the RAM config.
function Invoke-Bench {
    param(
        [string[]]$BenchArgs,   # llama-bench arguments (model, -ngl, -p, -n, etc.) - do NOT name this 'Args' (reserved)
        [string]$TestName       # e.g. 'cpu-gemma3' or 'hybrid-gptoss120b'
    )
    Assert-Bench
    $ram = Get-RamTag
    $cpu = Get-CpuInfo
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Cyan
    Write-Host " Test      : $TestName" -ForegroundColor Cyan
    Write-Host " RAM       : $($ram.TotalGB) GB  ($($ram.Count) sticks: $($ram.Sticks) GB) @ $($ram.SpeedMTs) MT/s" -ForegroundColor Cyan
    Write-Host " CPU       : $($cpu.Name)  ($($cpu.Cores)C/$($cpu.Threads)T)" -ForegroundColor Cyan
    Write-Host " When      : $stamp" -ForegroundColor Cyan
    Write-Host "==================================================================" -ForegroundColor Cyan
    Write-Host " > llama-bench $($BenchArgs -join ' ')" -ForegroundColor DarkGray
    Write-Host ""

    # Run llama-bench ONCE with CSV output (a second -o md run would reload the whole
    # model again - painful for 27B+ models). We parse the CSV, render a readable
    # summary table ourselves, and also append the raw rows to the results file.
    #
    # llama-bench writes an informational banner (ggml_cuda_init, load_backend) to
    # stderr. Under $ErrorActionPreference='Stop' + stream redirection that banner
    # would be promoted to a terminating error, so we drop to 'Continue' and judge
    # success/failure ONLY by $LASTEXITCODE (the authoritative signal).
    $eap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $csvOut = & $Bench @BenchArgs -o csv 2>$null
        $code = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $eap }
    if ($code -ne 0) { throw "llama-bench exited with code $code" }

    $lines = $csvOut -split "`r?`n" | Where-Object { $_ -ne '' }
    if ($lines.Count -lt 2) { throw "llama-bench produced no data rows." }

    # --- readable summary table rendered from the CSV -------------------------
    $rows = $lines | ConvertFrom-Csv
    Write-Host (" {0,-26} {1,4} {2,-9} {3,14}" -f 'model','ngl','test','t/s (avg)') -ForegroundColor White
    Write-Host (" {0}" -f ('-' * 58)) -ForegroundColor DarkGray
    foreach ($r in $rows) {
        $test = if ([int]$r.n_gen -gt 0) { "tg$($r.n_gen)" } else { "pp$($r.n_prompt)" }
        $avg  = [double]$r.avg_ts; $sd = [double]$r.stddev_ts
        Write-Host (" {0,-26} {1,4} {2,-9} {3,8:N2} +/- {4,4:N2}" -f `
            $r.model_type, $r.n_gpu_layers, $test, $avg, $sd)
    }

    # --- append raw rows (with our RAM metadata columns) to results\<test>_<ram>.csv
    $csvPath = Join-Path $ResultsDir "${TestName}_$($ram.Tag).csv"
    $exists  = Test-Path $csvPath
    $meta    = "$stamp,$($ram.TotalGB),$($ram.Sticks),$($ram.SpeedMTs)"
    $metaHdr = "run_time,ram_total_gb,ram_sticks_gb,ram_speed_mts"
    if (-not $exists) { "$metaHdr,$($lines[0])" | Out-File -FilePath $csvPath -Encoding utf8 }
    foreach ($d in ($lines | Select-Object -Skip 1)) { "$meta,$d" | Out-File -FilePath $csvPath -Append -Encoding utf8 }

    Write-Host ""
    Write-Host " Results appended -> $csvPath" -ForegroundColor Green
}

# Runs llama-cli for a fixed prompt/generation workload and logs per-repetition
# timing (prompt eval + eval tokens/sec) to results\<test>_<ram>.csv.
function Invoke-CliTimingBench {
    param(
        [string]$ModelPath,
        [string]$TestName,
        [int]$PromptTokens = 512,
        [int]$GenTokens = 128,
        [int]$Threads,
        [int]$Reps = 3,
        [string[]]$ExtraArgs
    )

    Assert-Cli

    $ram = Get-RamTag
    $cpu = Get-CpuInfo
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $modelLabel = [IO.Path]::GetFileNameWithoutExtension($ModelPath)
    $csvPath = Join-Path $ResultsDir "${TestName}_$($ram.Tag).csv"

    if (-not $Threads) { $Threads = (Get-CpuInfo).Cores }

    $promptText = @'
Repeat exactly this line once and stop.
'@

    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Cyan
    Write-Host " Test      : $TestName" -ForegroundColor Cyan
    Write-Host " Model     : $modelLabel" -ForegroundColor Cyan
    Write-Host " RAM       : $($ram.TotalGB) GB  ($($ram.Count) sticks: $($ram.Sticks) GB) @ $($ram.SpeedMTs) MT/s" -ForegroundColor Cyan
    Write-Host " CPU       : $($cpu.Name)  ($($cpu.Cores)C/$($cpu.Threads)T)" -ForegroundColor Cyan
    Write-Host " When      : $stamp" -ForegroundColor Cyan
    Write-Host "==================================================================" -ForegroundColor Cyan

    if (-not (Test-Path $csvPath)) {
        'run_time,ram_total_gb,ram_sticks_gb,ram_speed_mts,rep,tool,model,test,threads,prompt_tokens,gen_tokens,prompt_tps,gen_tps,cli_args' |
            Out-File -FilePath $csvPath -Encoding utf8
    }

    for ($i = 1; $i -le $Reps; $i++) {
        $args = @(
            '-m', $ModelPath,
            '-p', $promptText,
            '-n', $GenTokens,
            '--keep', '0',
            '-c', $PromptTokens,
            '-t', $Threads,
            '--no-warmup',
            '--simple-io',
            '--single-turn',
            '--no-display-prompt'
        )
        if ($ExtraArgs) { $args += $ExtraArgs }

        Write-Host ""
        Write-Host (">>> Rep {0}/{1}" -f $i, $Reps) -ForegroundColor Magenta
        Write-Host " > llama-cli $($args -join ' ')" -ForegroundColor DarkGray

        $eap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $out = & $Cli @args 2>&1
            $code = $LASTEXITCODE
        }
        finally { $ErrorActionPreference = $eap }

        if ($code -ne 0) {
            throw "llama-cli exited with code $code during repetition $i"
        }

        $joined = ($out | Out-String)
        $outputLines = $joined -split "`r?`n"
        $promptLine = $outputLines | Where-Object { $_ -match 'prompt eval time\s*=.*tokens per second' } | Select-Object -Last 1
        $evalLine = $outputLines | Where-Object { $_ -match '^eval time\s*=.*tokens per second' } | Select-Object -Last 1
        $summaryLine = $outputLines | Where-Object { $_ -match '\[\s*Prompt:\s*[0-9]+(?:\.[0-9]+)?\s+t/s\s*\|\s*Generation:\s*[0-9]+(?:\.[0-9]+)?\s+t/s\s*\]' } | Select-Object -Last 1

        $promptTps = [double]::NaN
        $genTps = [double]::NaN
        if ($promptLine -and $promptLine -match '([0-9]+(?:\.[0-9]+)?)\s+tokens per second') {
            $promptTps = [double]$Matches[1]
        }
        if ($evalLine -and $evalLine -match '([0-9]+(?:\.[0-9]+)?)\s+tokens per second') {
            $genTps = [double]$Matches[1]
        }
        if ($summaryLine -and $summaryLine -match 'Prompt:\s*([0-9]+(?:\.[0-9]+)?)\s+t/s\s*\|\s*Generation:\s*([0-9]+(?:\.[0-9]+)?)\s+t/s') {
            if ([double]::IsNaN($promptTps)) {
                $promptTps = [double]$Matches[1]
            }
            if ([double]::IsNaN($genTps)) {
                $genTps = [double]$Matches[2]
            }
        }

        if ([double]::IsNaN($genTps)) {
            throw "Failed to parse eval throughput from llama-cli output on repetition $i"
        }

        $promptDisplay = if ([double]::IsNaN($promptTps)) { 'n/a' } else { '{0:N2}' -f $promptTps }
        Write-Host (" prompt t/s: {0}" -f $promptDisplay) -ForegroundColor White
        Write-Host (" gen t/s   : {0:N2}" -f $genTps) -ForegroundColor White

        $flatArgs = (($args | ForEach-Object { ([string]$_).Replace(',', ';') }) -join ' ')
        $promptCsv = if ([double]::IsNaN($promptTps)) { '' } else { '{0:F6}' -f $promptTps }
        $genCsv = '{0:F6}' -f $genTps
        $row = "{0},{1},{2},{3},{4},llama-cli,{5},{6},{7},{8},{9},{10},{11},{12}" -f `
            $stamp, $ram.TotalGB, $ram.Sticks, $ram.SpeedMTs, $i, $modelLabel, $TestName, $Threads, $PromptTokens, $GenTokens, `
            $promptCsv, $genCsv, $flatArgs
        $row | Out-File -FilePath $csvPath -Append -Encoding utf8
    }

    Write-Host ""
    Write-Host " Results appended -> $csvPath" -ForegroundColor Green
}
