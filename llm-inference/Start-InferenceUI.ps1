<#
.SYNOPSIS
  Local web playground to test inference against the llama-server started by Serve-Qwen.ps1.
  Shows per-request stats: time-to-first-token (TTFT), prompt processing t/s, generation t/s.

.DESCRIPTION
  Serves a small self-contained HTML page on http://localhost:<UiPort> and opens it in the
  browser. The page streams chat completions from <ServerUrl>/v1/chat/completions and reports
  timings (server-authoritative when available, else client-measured).

  Must run locally (talks to your local llama-server); cannot be a hosted artifact.

.PARAMETER ServerUrl
  Base URL of the llama-server. Default http://localhost:8081 (Serve-Qwen.ps1 runs there,
  because port 8080 on this box is taken by the AgentService process).

.PARAMETER UiPort
  Port for this test UI. Default 8082.
  NOTE: llama-server also serves its own full chat UI on the SAME port as the API
  (http://localhost:8081) - this script is the lightweight timing playground (TTFT / prefill
  / decode stats per request), not a replacement for it.

.PARAMETER ApiKey
  API key, if the llama-server was started with -ApiKey.

.EXAMPLE
  .\Start-InferenceUI.ps1
  .\Start-InferenceUI.ps1 -ServerUrl http://localhost:9000 -ApiKey "my-secret"
#>
param(
    [string]$ServerUrl = 'http://localhost:8081',
    [int]$UiPort = 8082,
    [string]$ApiKey = ''
)

$ErrorActionPreference = 'Stop'

$html = @'
<!doctype html>
<html lang="sk">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen Inference Test</title>
<style>
  :root{
    --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --line:#2a2f3a; --tx:#e6e8ee;
    --mut:#9aa3b2; --acc:#5b9dff; --ok:#37d39a; --warn:#f5b74e; --err:#ff6b6b;
  }
  @media (prefers-color-scheme:light){
    :root{ --bg:#f4f6fa; --panel:#fff; --panel2:#eef1f6; --line:#dce1ea; --tx:#141821;
      --mut:#5a6472; --acc:#2a6df0; }
  }
  *{box-sizing:border-box}
  body{margin:0;font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--tx)}
  .wrap{max-width:920px;margin:0 auto;padding:18px}
  h1{font-size:18px;margin:0 0 2px}
  .sub{color:var(--mut);font-size:13px;margin-bottom:14px}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--mut);margin-right:6px;vertical-align:middle}
  .dot.ok{background:var(--ok)} .dot.err{background:var(--err)}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:14px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
  label{display:block;font-size:12px;color:var(--mut);margin-bottom:4px}
  input,textarea,select{width:100%;background:var(--panel2);border:1px solid var(--line);color:var(--tx);
    border-radius:8px;padding:8px;font:inherit}
  textarea{resize:vertical;min-height:90px}
  .row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px}
  button{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:10px 18px;font:inherit;
    font-weight:600;cursor:pointer}
  button.sec{background:var(--panel2);color:var(--tx);border:1px solid var(--line)}
  button:disabled{opacity:.5;cursor:not-allowed}
  .chk{display:flex;align-items:center;gap:6px;font-size:13px;color:var(--mut)}
  .chk input{width:auto}
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}
  .stat{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
  .stat .k{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
  .stat .v{font-size:22px;font-weight:700;margin-top:2px}
  .stat .u{font-size:12px;color:var(--mut);font-weight:500}
  .out{white-space:pre-wrap;word-break:break-word;min-height:60px}
  details{margin-bottom:10px} summary{cursor:pointer;color:var(--mut);font-size:13px}
  .think{white-space:pre-wrap;color:var(--mut);font-size:13px;border-left:2px solid var(--line);padding-left:10px;margin-top:8px}
  .src{color:var(--mut);font-size:12px}
</style>
</head>
<body>
<div class="wrap">
  <h1>Qwen Inference Test</h1>
  <div class="sub"><span id="dot" class="dot"></span><span id="srvtxt">kontrolujem server...</span>
    &nbsp;·&nbsp; <span class="src">endpoint: <code id="ep"></code></span></div>

  <div class="panel">
    <div class="grid">
      <div><label>Server URL</label><input id="server"></div>
      <div><label>API key (voliteľné)</label><input id="apikey" placeholder="ak je nastavený"></div>
      <div><label>Max tokenov</label><input id="maxtok" type="number" value="256" min="1"></div>
      <div><label>Teplota</label><input id="temp" type="number" value="0.7" step="0.1" min="0"></div>
    </div>
    <div class="row"><div style="flex:1"><label>System prompt (voliteľné)</label>
      <input id="system" placeholder="napr. Si nápomocný asistent."></div></div>
    <div class="row"><div style="flex:1"><label>Prompt</label>
      <textarea id="prompt">Vysvetli v troch bodoch, preco je more slane.</textarea></div></div>
    <div class="row">
      <button id="send">Odoslať</button>
      <button id="stop" class="sec" disabled>Stop</button>
      <label class="chk"><input type="checkbox" id="think"> Reasoning (thinking)</label>
    </div>
  </div>

  <div class="panel">
    <div class="stats">
      <div class="stat"><div class="k">Čas do 1. tokenu</div><div class="v"><span id="s_ttft">–</span><span class="u"> ms</span></div></div>
      <div class="stat"><div class="k">Prompt processing</div><div class="v"><span id="s_pp">–</span><span class="u"> t/s</span></div></div>
      <div class="stat"><div class="k">Token generation</div><div class="v"><span id="s_tg">–</span><span class="u"> t/s</span></div></div>
      <div class="stat"><div class="k">Celkový čas</div><div class="v"><span id="s_tot">–</span><span class="u"> s</span></div></div>
      <div class="stat"><div class="k">Prompt / gen tokenov</div><div class="v" style="font-size:18px"><span id="s_ntok">–</span></div></div>
    </div>
  </div>

  <div class="panel">
    <details id="thinkbox" style="display:none"><summary>Reasoning / thinking</summary>
      <div id="think_out" class="think"></div></details>
    <div id="out" class="out"></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
$('server').value = '__SERVER_URL__';
$('apikey').value = '__API_KEY__';
$('ep').textContent = '__SERVER_URL__/v1/chat/completions';

let controller = null;

async function ping(){
  try{
    const r = await fetch($('server').value.replace(/\/$/,'') + '/health', {cache:'no-store'});
    const ok = r.ok;
    $('dot').className = 'dot ' + (ok?'ok':'err');
    $('srvtxt').textContent = ok ? 'server pripojený' : 'server neodpovedá (HTTP '+r.status+')';
  }catch(e){
    $('dot').className='dot err'; $('srvtxt').textContent='server nedostupný — beží Serve-Qwen.ps1?';
  }
}
ping(); setInterval(ping, 5000);

function setBusy(b){ $('send').disabled=b; $('stop').disabled=!b; }
function fmt(n,d=1){ return (n==null||isNaN(n))?'–':Number(n).toFixed(d); }

async function run(){
  const base = $('server').value.replace(/\/$/,'');
  const key = $('apikey').value.trim();
  const body = {
    messages: [],
    max_tokens: parseInt($('maxtok').value)||256,
    temperature: parseFloat($('temp').value),
    stream: true,
    stream_options: { include_usage: true },
    chat_template_kwargs: { enable_thinking: $('think').checked }
  };
  if ($('system').value.trim()) body.messages.push({role:'system', content:$('system').value});
  body.messages.push({role:'user', content:$('prompt').value});

  $('out').textContent=''; $('think_out').textContent='';
  $('thinkbox').style.display = $('think').checked ? 'block':'none';
  ['s_ttft','s_pp','s_tg','s_tot','s_ntok'].forEach(i=>$(i).textContent='…');
  setBusy(true);
  controller = new AbortController();

  const t0 = performance.now();
  let tFirst = null, timings = null, usage = null, gotText='';
  try{
    const headers = {'Content-Type':'application/json'};
    if (key) headers['Authorization'] = 'Bearer '+key;
    const resp = await fetch(base+'/v1/chat/completions', {method:'POST',headers,body:JSON.stringify(body),signal:controller.signal});
    if (!resp.ok){ $('out').textContent = 'Chyba HTTP '+resp.status+': '+await resp.text(); setBusy(false); return; }
    const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf='';
    while(true){
      const {value,done} = await reader.read(); if(done) break;
      buf += dec.decode(value,{stream:true});
      let idx;
      while((idx = buf.indexOf('\n')) >= 0){
        const line = buf.slice(0,idx).trim(); buf = buf.slice(idx+1);
        if(!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if(data === '[DONE]') continue;
        let obj; try{ obj = JSON.parse(data); }catch{ continue; }
        if(obj.timings) timings = obj.timings;
        if(obj.usage) usage = obj.usage;
        const d = obj.choices && obj.choices[0] && obj.choices[0].delta;
        if(d){
          if(d.reasoning_content){ if(tFirst==null)tFirst=performance.now(); $('think_out').textContent += d.reasoning_content; }
          if(d.content){ if(tFirst==null)tFirst=performance.now(); gotText += d.content; $('out').textContent = gotText; }
        }
      }
    }
  }catch(e){
    if(e.name!=='AbortError') $('out').textContent += '\n[chyba: '+e.message+']';
  }
  const tEnd = performance.now();

  // TTFT: client-measured wall clock to first token.
  const ttft = tFirst!=null ? (tFirst - t0) : null;
  const totalS = (tEnd - t0)/1000;
  // Prefer server-authoritative timings; fall back to client math.
  let pp=null, tg=null, pn=null, cn=null;
  if(timings){
    pp = timings.prompt_per_second; tg = timings.predicted_per_second;
    pn = timings.prompt_n; cn = timings.predicted_n;
  }
  if(usage){ pn = pn ?? usage.prompt_tokens; cn = cn ?? usage.completion_tokens; }
  if(pp==null && pn && ttft) pp = pn / (ttft/1000);
  if(tg==null && cn && tFirst!=null) tg = cn / ((tEnd - tFirst)/1000);

  $('s_ttft').textContent = ttft!=null ? Math.round(ttft) : '–';
  $('s_pp').textContent = fmt(pp,1);
  $('s_tg').textContent = fmt(tg,2);
  $('s_tot').textContent = fmt(totalS,1);
  $('s_ntok').textContent = (pn??'?') + ' / ' + (cn??'?');
  setBusy(false);
}

$('send').onclick = run;
$('stop').onclick = ()=>{ if(controller) controller.abort(); setBusy(false); };
$('prompt').addEventListener('keydown', e=>{ if(e.ctrlKey && e.key==='Enter') run(); });
</script>
</body>
</html>
'@

$html = $html.Replace('__SERVER_URL__', $ServerUrl.TrimEnd('/')).Replace('__API_KEY__', $ApiKey)

$prefix = "http://localhost:$UiPort/"
$listener = [System.Net.HttpListener]::new()
$listener.Prefixes.Add($prefix)
try { $listener.Start() }
catch { throw "Nepodarilo sa otvoriť $prefix  ($($_.Exception.Message)). Skús iný -UiPort." }

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Inference Test UI beží na:  $prefix" -ForegroundColor Green
Write-Host " Napojené na llama-server :  $ServerUrl" -ForegroundColor Gray
Write-Host " (Ctrl+C tu ukončí testovacie UI. llama-server tým NEZASTAVÍŠ.)" -ForegroundColor DarkGray
Write-Host "==================================================================" -ForegroundColor Cyan

Start-Process $prefix | Out-Null

try {
    while ($listener.IsListening) {
        $ctx = $listener.GetContext()
        $path = $ctx.Request.Url.AbsolutePath
        $resp = $ctx.Response
        if ($path -eq '/favicon.ico') { $resp.StatusCode = 204; $resp.Close(); continue }
        $buf = [Text.Encoding]::UTF8.GetBytes($html)
        $resp.ContentType = 'text/html; charset=utf-8'
        $resp.Headers['Cache-Control'] = 'no-store'
        $resp.ContentLength64 = $buf.Length
        $resp.OutputStream.Write($buf, 0, $buf.Length)
        $resp.OutputStream.Close()
    }
} finally {
    $listener.Stop(); $listener.Close()
}
