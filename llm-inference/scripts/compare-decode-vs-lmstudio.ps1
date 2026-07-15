# Head-to-head decode (token generation) speed: our ROCm 7 llama-server vs LM Studio.
# Same short prompt + temperature 0 (deterministic => same output => fair MTP comparison).
# Short prompt => prefill time negligible, so decode_tps ~= completion_tokens / wall_time
# (we still subtract an estimated prefill using the measured ~265 t/s prefill rate).
$ErrorActionPreference = 'Continue'
$PREFILL_TPS = 265.0

$servers = @(
    @{ name='NAS (ROCm7)  :8080'; url='http://localhost:8080' },
    @{ name='LM Studio    :1234'; url='http://localhost:1234' }
)

$body = @{
    messages = @(@{ role='user'; content='Vysvetli podrobne, ako funguje fotosyntéza. Odpovedz v aspon 8 vetach.' })
    max_tokens = 256
    temperature = 0
    stream = $false
} | ConvertTo-Json -Depth 5 -Compress

foreach ($s in $servers) {
    Write-Host "`n==================== $($s.name) ====================" -ForegroundColor Yellow
    try { $m = Invoke-RestMethod "$($s.url)/v1/models" -TimeoutSec 5 -ErrorAction Stop; $mid=$m.data[0].id }
    catch { Write-Host "  NEDOSTUPNY ($($_.Exception.Message))" -ForegroundColor Red; continue }
    Write-Host "  model: $([IO.Path]::GetFileName($mid))" -ForegroundColor Gray
    try {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $r = Invoke-RestMethod "$($s.url)/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 300 -ErrorAction Stop
        $sw.Stop()
        $ptok = $r.usage.prompt_tokens
        $ctok = $r.usage.completion_tokens
        $wall = $sw.Elapsed.TotalSeconds
        $prefillS = $ptok / $PREFILL_TPS
        $decodeS = [Math]::Max($wall - $prefillS, 0.001)
        $dtps = $ctok / $decodeS
        Write-Host ("  prompt/gen tok : {0} / {1}" -f $ptok,$ctok)
        Write-Host ("  wall / decode  : {0:N1}s / {1:N1}s" -f $wall,$decodeS)
        Write-Host ("  TOKEN GENERATION: {0:N2} t/s" -f $dtps) -ForegroundColor Green
        if ($r.timings.predicted_per_second) {
            Write-Host ("  (server timings: {0:N2} t/s)" -f $r.timings.predicted_per_second) -ForegroundColor DarkGray
        }
    } catch { Write-Host "  chyba: $($_.Exception.Message)" -ForegroundColor Red }
}
Write-Host "`nDONE" -ForegroundColor Green
