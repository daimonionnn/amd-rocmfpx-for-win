# Head-to-head prompt-processing (prefill) speed: our ROCm 7 llama-server vs LM Studio.
# Same long prompt to both, max_tokens=1 so wall time ~= prefill time. Apples-to-apples,
# both measured client-side (prompt_tokens / wall_seconds).
$ErrorActionPreference = 'Continue'
$sp = 'C:\Users\matt\AppData\Local\Temp\claude\c--development-ai-tools-for-win\4d6c6f75-de03-4360-b40b-5d2c76574ddc\scratchpad'
$prompt = Get-Content -Raw "$sp\prompt-32k.txt"

$servers = @(
    @{ name='NAS (ROCm7)  :8080'; url='http://localhost:8080' },
    @{ name='LM Studio    :1234'; url='http://localhost:1234' }
)

$body = @{ messages=@(@{role='user';content=$prompt}); max_tokens=1; temperature=0; stream=$false } |
        ConvertTo-Json -Depth 5 -Compress

foreach ($s in $servers) {
    Write-Host "`n==================== $($s.name) ====================" -ForegroundColor Yellow
    try { $m = Invoke-RestMethod "$($s.url)/v1/models" -TimeoutSec 5 -ErrorAction Stop; $mid = $m.data[0].id }
    catch { Write-Host "  NEDOSTUPNY ($($_.Exception.Message))" -ForegroundColor Red; continue }
    Write-Host "  model: $([IO.Path]::GetFileName($mid))" -ForegroundColor Gray
    Write-Host "  posielam ~34.5K prompt (max_tokens=1)..." -ForegroundColor DarkGray
    try {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $r = Invoke-RestMethod "$($s.url)/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600 -ErrorAction Stop
        $sw.Stop()
        $ptok = $r.usage.prompt_tokens
        $secs = $sw.Elapsed.TotalSeconds
        $pps  = if ($secs -gt 0) { $ptok / $secs } else { 0 }
        Write-Host ("  prompt tokenov : {0}" -f $ptok)
        Write-Host ("  cas (prefill)  : {0:N1} s" -f $secs)
        Write-Host ("  PROMPT PROCESSING: {0:N1} t/s" -f $pps) -ForegroundColor Green
        if ($r.timings.prompt_per_second) {
            Write-Host ("  (server timings: {0:N1} t/s)" -f $r.timings.prompt_per_second) -ForegroundColor DarkGray
        }
    } catch { Write-Host "  chyba pri generovani: $($_.Exception.Message)" -ForegroundColor Red }
}
Write-Host "`nDONE" -ForegroundColor Green
