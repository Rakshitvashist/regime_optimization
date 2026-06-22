"""
vol_server.py  —  live volatility dashboard (premium React frontend + Python backend).

Backend: stdlib http.server (no Flask). Computes the full dashboard data on
startup, caches it, refreshes in a background thread every TTL. Serves:
    GET /        -> the React app (Obsidian-themed; React18 + Babel + Plotly via CDN)
    GET /data    -> cached JSON
    GET /health  -> status

The same React app powers the static GitHub-Pages build (data embedded via
window.__PRELOAD__; see build_static_dashboard.py).

Run:
    python vol_server.py --port 8000 [--garch] [--ttl 900]
    -> http://localhost:8000
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import vol_dashboard_data as vdd

_CACHE = {"data": {}, "ts": 0, "computing": False}
_GARCH = False


def _recompute():
    _CACHE["computing"] = True
    try:
        d = vdd.compute_all(garch=_GARCH, log=lambda m: print(m, flush=True))
        _CACHE["data"] = d
        _CACHE["ts"] = time.time()
    finally:
        _CACHE["computing"] = False


def _bg_loop(ttl):
    while True:
        time.sleep(ttl)
        print("[recompute] refreshing dashboard data...", flush=True)
        try:
            _recompute()
        except Exception as e:
            print(f"[recompute] failed: {e}", flush=True)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="text/html; charset=utf-8", code=200):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/data"):
            self._send(json.dumps({"data": _CACHE["data"], "ts": _CACHE["ts"],
                                   "computing": _CACHE["computing"]}), "application/json")
        elif self.path.startswith("/health"):
            self._send(json.dumps({"ok": True}), "application/json")
        else:
            self._send(PAGE)


def main():
    global _GARCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ttl", type=int, default=900)
    ap.add_argument("--garch", action="store_true")
    a = ap.parse_args()
    _GARCH = a.garch
    print("Computing dashboard data (first run ~1-2 min)...", flush=True)
    _recompute()
    threading.Thread(target=_bg_loop, args=(a.ttl,), daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    print(f"\nServing on http://localhost:{a.port}   (Ctrl+C to stop)", flush=True)
    srv.serve_forever()


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Volatility Intelligence</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{
 --bg:#0A0A0F;--surface:#111118;--surface2:#16161F;--line:#22222E;
 --accent:#6C63FF;--accent2:#00D4AA;--danger:#FF4757;--warn:#FFB020;
 --text:#F0F0FF;--muted:#8888AA;
 --tf:150ms;--tb:250ms;--ease:cubic-bezier(0.4,0,0.2,1);
 --mono:'JetBrains Mono',monospace;--ui:'Inter',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:var(--ui);font-size:14px;overflow-x:hidden}
::-webkit-scrollbar{width:9px;height:9px}
::-webkit-scrollbar-thumb{background:rgba(108,99,255,.5);border-radius:8px}
::-webkit-scrollbar-thumb:hover{background:var(--accent)}
::-webkit-scrollbar-track{background:transparent}
a{text-decoration:none}
.app{display:flex;min-height:100vh}
.sidebar{width:240px;position:fixed;inset:0 auto 0 0;background:var(--surface);border-right:1px solid var(--line);padding:22px 0;display:flex;flex-direction:column;z-index:30}
.brand{display:flex;align-items:center;gap:11px;padding:0 22px 24px;font-weight:800;font-size:16px;letter-spacing:.2px}
.brand .logo{width:32px;height:32px;border-radius:9px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:flex;align-items:center;justify-content:center;font-weight:800;color:#0A0A0F;box-shadow:0 6px 18px rgba(108,99,255,.35)}
.nav{display:flex;flex-direction:column;gap:3px;padding:0 12px;flex:1}
.nav a{position:relative;display:flex;align-items:center;gap:12px;padding:11px 14px;border-radius:11px;color:var(--muted);font-weight:500;border-left:2px solid transparent;overflow:hidden;will-change:transform;transition:color var(--tb) var(--ease),border-color var(--tb) var(--ease)}
.nav a::before{content:'';position:absolute;inset:0;background:linear-gradient(90deg,rgba(108,99,255,.20),transparent);transform:translateX(-100%);transition:transform var(--tb) var(--ease);z-index:0}
.nav a span,.nav a svg{position:relative;z-index:1}
.nav a:hover,.nav a.active{color:var(--text);border-left-color:var(--accent)}
.nav a:hover::before,.nav a.active::before{transform:translateX(0)}
.nav a svg{width:18px;height:18px;transition:transform var(--tb) var(--ease)}
.nav a:hover svg{transform:scale(1.15)}
.side-foot{padding:16px 22px 0;color:var(--muted);font-size:11px;border-top:1px solid var(--line);margin:8px 12px 0}
.main{margin-left:240px;flex:1;min-width:0}
.header{position:sticky;top:0;z-index:25;display:flex;align-items:center;justify-content:space-between;gap:16px;padding:14px 28px;background:rgba(17,17,24,.7);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border-bottom:1px solid var(--line)}
.header h1{font-size:17px;font-weight:700} .header .sub{color:var(--muted);font-size:12px;margin-top:2px}
.hgroup{display:flex;align-items:center;gap:14px}
select.inst{background:var(--surface2);color:var(--text);border:1px solid var(--line);padding:9px 15px;border-radius:11px;font-family:var(--ui);font-weight:600;font-size:13px;cursor:pointer;transition:border-color var(--tb) var(--ease),box-shadow var(--tb) var(--ease)}
select.inst:hover{border-color:var(--accent);box-shadow:0 0 0 3px rgba(108,99,255,.14)}
.live{display:flex;align-items:center;gap:7px;color:var(--accent2);font-size:12px;font-weight:600}
.pulse{width:8px;height:8px;border-radius:50%;background:var(--accent2);animation:pulse 2s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(0,212,170,.5)}70%{box-shadow:0 0 0 8px rgba(0,212,170,0)}100%{box-shadow:0 0 0 0 rgba(0,212,170,0)}}
.iconbtn{width:38px;height:38px;border-radius:11px;background:var(--surface2);border:1px solid var(--line);display:flex;align-items:center;justify-content:center;cursor:pointer;color:var(--muted);transition:transform var(--tb) var(--ease),border-color var(--tb) var(--ease),color var(--tb) var(--ease)}
.bell:hover{transform:rotate(14deg);border-color:var(--accent);color:var(--text)}
.avatar{width:38px;height:38px;border-radius:50%;background:linear-gradient(135deg,var(--accent),var(--accent2));display:flex;align-items:center;justify-content:center;font-weight:700;color:#0A0A0F;cursor:pointer;transition:transform var(--tb) var(--ease)}
.avatar:hover{transform:scale(1.08)}
.content{padding:24px 28px 70px;max-width:1280px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:22px}
.kpi{background:var(--surface);border:1px solid transparent;border-radius:16px;padding:18px 20px;position:relative;overflow:hidden;will-change:transform;opacity:0;transform:translateY(12px);animation:rise .55s var(--ease) forwards;transition:transform var(--tb) var(--ease),box-shadow var(--tb) var(--ease),border-color var(--tb) var(--ease)}
.kpi:hover{transform:translateY(-4px);box-shadow:0 20px 40px rgba(108,99,255,.15);border-color:var(--accent)}
.kpi .k{color:var(--muted);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.6px}
.kpi .v{font-family:var(--mono);font-size:28px;font-weight:700;margin-top:11px;line-height:1}
.kpi .d{font-size:12px;font-weight:600;margin-top:9px;display:flex;align-items:center;gap:5px}
.kpi .spark{position:absolute;right:14px;top:16px;opacity:.5}
.up{color:var(--accent2)}.down{color:var(--danger)}.flat{color:var(--muted)}
@keyframes rise{to{opacity:1;transform:translateY(0)}}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin-bottom:16px;transition:border-color var(--tb) var(--ease)}
.panel:hover{border-color:rgba(108,99,255,.4)}
.panel h3{font-size:14px;font-weight:600;margin-bottom:14px;display:flex;align-items:center;gap:9px}
.panel h3 .tag{font-size:10px;color:var(--accent2);background:rgba(0,212,170,.12);padding:3px 9px;border-radius:20px;font-weight:700;letter-spacing:.4px}
.row2{display:grid;grid-template-columns:1.7fr 1fr;gap:16px}
.row2>.panel{margin-bottom:0}
table{border-collapse:collapse;width:100%;font-size:13px;font-family:var(--mono)}
th,td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:10px;text-transform:uppercase;font-family:var(--ui);letter-spacing:.5px}
td.l,th.l{text-align:left}
tbody tr{position:relative;transition:color var(--tf) var(--ease)}
tbody tr::after{content:'';position:absolute;inset:0;background:linear-gradient(90deg,rgba(108,99,255,.10),transparent);transform:scaleX(0);transform-origin:left;transition:transform var(--tb) var(--ease);z-index:-1}
tbody tr:hover::after{transform:scaleX(1)}
.badge{display:inline-block;padding:3px 11px;border-radius:20px;font-size:12px;font-weight:600;font-family:var(--ui)}
.b-quiet{background:rgba(0,212,170,.15);color:var(--accent2)}
.b-normal{background:rgba(255,176,32,.16);color:var(--warn)}
.b-explosive{background:rgba(255,71,87,.16);color:var(--danger)}
.b-na{background:var(--surface2);color:var(--muted)}
.gauge{height:8px;background:var(--surface2);border-radius:6px;overflow:hidden;margin-top:9px}
.gauge>div{height:100%;background:linear-gradient(90deg,var(--accent2),var(--warn),var(--danger));transition:width var(--tb) var(--ease)}
.chips{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}
.sub{color:var(--muted);font-size:12px}
.tabs{display:flex;gap:8px;overflow-x:auto;padding:16px 28px 2px;position:sticky;top:67px;z-index:14;background:rgba(10,10,15,.85);backdrop-filter:blur(10px)}
.tab{padding:9px 16px;border-radius:11px;background:var(--surface);border:1px solid var(--line);color:var(--muted);font-weight:600;font-size:13px;font-family:var(--ui);cursor:pointer;white-space:nowrap;transition:color var(--tb) var(--ease),border-color var(--tb) var(--ease),transform var(--tb) var(--ease),background var(--tb) var(--ease)}
.tab:hover{color:var(--text);border-color:var(--accent);transform:translateY(-2px)}
.tab.on{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#0A0A0F;border-color:transparent;box-shadow:0 8px 20px rgba(108,99,255,.3)}
@media(max-width:980px){.kpis{grid-template-columns:repeat(2,1fr)}.row2{grid-template-columns:1fr}
 .sidebar{width:62px}.brand span,.nav a span,.side-foot{display:none}.main{margin-left:62px}.content{padding:18px}}
</style></head><body><div id="root"></div>
<script type="text/plain" id="appsrc">
const {useState,useEffect,useRef}=React;
const FILL={2:'rgba(108,99,255,0.10)',1.5:'rgba(108,99,255,0.20)',1:'rgba(108,99,255,0.38)'};
const fmt=n=>n==null||isNaN(n)?'—':(Math.abs(n)>=1000?Number(n).toLocaleString(undefined,{maximumFractionDigits:0}):Number(n).toFixed(2));
const IC={
 dash:<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>,
 chart:<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 3v18h18"/><path d="M7 13l3-3 3 2 5-6"/></svg>,
 layers:<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2 2 7l10 5 10-5z"/><path d="M2 17l10 5 10-5M2 12l10 5 10-5"/></svg>,
 tag:<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20.6 13.4 12 22l-9-9V3h10z"/><circle cx="7.5" cy="7.5" r="1.5"/></svg>,
 shield:<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2 4 5v6c0 5 3.4 8.5 8 11 4.6-2.5 8-6 8-11V5z"/><path d="m9 12 2 2 4-4"/></svg>,
 grid:<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="18" r="2.5"/><path d="M8 6h8M6 8v8M18 8v8M8 18h8"/></svg>,
 bell:<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 8a6 6 0 1 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.7 21a2 2 0 0 1-3.4 0"/></svg>,
};
const PLOT={paper_bgcolor:'#111118',plot_bgcolor:'#111118',font:{color:'#F0F0FF',family:'Inter'},margin:{t:8,r:10,b:42,l:48}};
function Badge({r}){return <span className={'badge '+(({quiet:'b-quiet',normal:'b-normal',explosive:'b-explosive'})[r]||'b-na')}>{r}</span>;}

function Cone({inst}){const ref=useRef();
 useEffect(()=>{if(!inst||!window.Plotly)return;const c=inst.forecast_cone,H=c.map(r=>r.H);let tr=[];
  [2,1.5,1].forEach(k=>{tr.push({x:H,y:c.map(r=>r['up'+k]),mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'});
   tr.push({x:H,y:c.map(r=>r['dn'+k]),mode:'lines',line:{width:0},fill:'tonexty',fillcolor:FILL[k],name:k+'σ',hoverinfo:'skip'});});
  tr.push({x:H,y:c.map(r=>r.median),mode:'lines+markers',line:{color:'#6C63FF',width:3},marker:{size:6,color:'#6C63FF'},name:'HAR median'});
  const hc=inst.hist_cone||[];if(hc.length){tr.push({x:hc.map(r=>r.H),y:hc.map(r=>r.median),mode:'lines',line:{color:'#00D4AA',width:1.6,dash:'dot'},name:'hist median'});}
  if(inst.garch)tr.push({x:inst.garch.map(r=>r.H),y:inst.garch.map(r=>r.median),mode:'lines',line:{color:'#FFB020',width:2,dash:'dash'},name:'GARCH'});
  window.Plotly.react(ref.current,tr,{...PLOT,xaxis:{title:'horizon (days)',gridcolor:'#22222E',zeroline:false},yaxis:{title:'ann vol %',gridcolor:'#22222E',zeroline:false},legend:{orientation:'h',font:{size:11}},hoverlabel:{bgcolor:'#16161F',bordercolor:'#6C63FF'}},{displayModeBar:false,responsive:true});
 },[inst]);return <div ref={ref} style={{height:420}}/>;}
function Hist({inst}){const ref=useRef();
 useEffect(()=>{if(!inst||!window.Plotly)return;
  window.Plotly.react(ref.current,[{x:inst.hist_series.date,y:inst.hist_series.vol,mode:'lines',line:{color:'#00D4AA',width:1.3},fill:'tozeroy',fillcolor:'rgba(0,212,170,0.06)'}],
   {...PLOT,xaxis:{gridcolor:'#22222E'},yaxis:{title:'ann vol %',gridcolor:'#22222E'}},{displayModeBar:false,responsive:true});
 },[inst]);return <div ref={ref} style={{height:240}}/>;}
function Corr({corr}){const ref=useRef();
 useEffect(()=>{if(!corr||!window.Plotly)return;const L=corr.labels;
  window.Plotly.react(ref.current,[{z:corr.matrix,x:L,y:L,type:'heatmap',zmin:-1,zmax:1,
   colorscale:[[0,'#FF4757'],[0.5,'#0A0A0F'],[1,'#00D4AA']],text:corr.matrix,texttemplate:'%{text}',textfont:{size:10,family:'JetBrains Mono'},showscale:true,colorbar:{thickness:10,len:.7}}],
   {...PLOT,margin:{t:8,r:8,b:96,l:96},xaxis:{tickangle:-40},yaxis:{autorange:'reversed'}},{displayModeBar:false,responsive:true});
 },[corr]);if(!corr)return null;
 return <div><div ref={ref} style={{height:400}}/><div className="chips">
   {corr.top.map((p,i)=><span key={i} className="badge" style={{background:p[2]>0?'rgba(0,212,170,.15)':'rgba(255,71,87,.15)',color:p[2]>0?'var(--accent2)':'var(--danger)'}}>{p[0]}–{p[1]} {p[2]}</span>)}</div></div>;}
function BandTable({bands,price,title}){return <div style={{flex:1,minWidth:300}}><div className="sub" style={{marginBottom:6}}>{title}</div>
 <table><thead><tr><th className="l">H</th><th>-2σ</th><th>-1σ</th><th>price</th><th>+1σ</th><th>+2σ</th></tr></thead>
  <tbody>{bands.map(b=><tr key={b.H}><td className="l">{b.H}d</td><td className="down">{fmt(b.dn2)}</td><td>{fmt(b.dn1)}</td>
    <td style={{color:'#6C63FF',fontWeight:700}}>{fmt(price)}</td><td>{fmt(b.up1)}</td><td className="up">{fmt(b.up2)}</td></tr>)}</tbody></table></div>;}
function Reliability({cov}){if(!cov)return null;const rows=[];const ms=['raw','drift','tuned'];const lab={raw:'plain',drift:'+drift',tuned:'auto-tuned'};const col={raw:'#8888AA',drift:'#00D4AA',tuned:'#6C63FF'};
 Object.keys(cov).forEach(H=>ms.forEach(m=>{if(cov[H][m])rows.push([H,m,cov[H][m]])}));
 const cell=(val,tgt)=>{const p=val*100;const off=Math.abs(p-tgt);const c=off<=4?'#00D4AA':off<=9?'#E8B339':'#FF4757';return <td style={{color:c,fontWeight:off<=4?700:400}}>{p.toFixed(0)}%</td>;};
 return <table><thead><tr><th className="l">H</th><th className="l">band</th><th>1σ <span className="sub">·68</span></th><th>1.5σ <span className="sub">·87</span></th><th>2σ <span className="sub">·95</span></th></tr></thead>
  <tbody>{rows.map(([H,m,d],i)=><tr key={i}><td className="l">{m==='raw'?H+'d':''}</td><td className="l" style={{color:col[m],fontWeight:m==='tuned'?700:400}}>{lab[m]}</td>
   {cell(d['1'],68)}{cell(d['1.5'],87)}{cell(d['2'],95)}</tr>)}</tbody></table>;}

function StatBox({k,v,d,tone}){return <div style={{flex:1,minWidth:130,background:'var(--surface2)',border:'1px solid var(--line)',borderRadius:12,padding:'12px 14px'}}>
 <div className="sub" style={{fontSize:11,textTransform:'uppercase',letterSpacing:'.5px'}}>{k}</div>
 <div style={{fontFamily:'var(--mono)',fontSize:20,fontWeight:700,marginTop:6,color:tone||'var(--text)'}}>{v}</div>
 {d&&<div className="sub" style={{marginTop:3}}>{d}</div>}</div>;}
function StatePanel({s}){ if(!s||s.error) return <div className="sub">{(s&&s.error)||'n/a'}</div>;
 const tt=s.trend==='trending'?'var(--accent2)':s.trend==='mean-revert'?'var(--warn)':'var(--muted)';
 const ft=s.tail==='fat-tailed'?'var(--danger)':'var(--accent2)';
 return <div style={{display:'flex',gap:12,flexWrap:'wrap'}}>
  <StatBox k="Trend (Hurst)" v={s.hurst} d={s.trend} tone={tt}/>
  <StatBox k="Tail (excess kurt)" v={s.kurtosis} d={s.tail} tone={ft}/>
  <StatBox k="Tail index α (EVT)" v={s.tail_index} d="lower = fatter tail"/>
  <StatBox k="Skew" v={s.skew} d={s.skew<0?'left / crash skew':'right skew'} tone={s.skew<0?'var(--danger)':'var(--text)'}/>
  <StatBox k="Daily VaR 95%" v={s.var95+'%'} d={'CVaR '+s.cvar95+'%'} tone="var(--danger)"/>
  <StatBox k="Drawdown" v={s.cur_dd+'%'} d={'max '+s.max_dd+'%'} tone="var(--danger)"/>
 </div>;}
function TomorrowPanel({t}){ if(!t||t.error) return <div className="sub">{(t&&t.error)||'n/a'}</div>;
 const calmTone=t.p_calm_tomorrow>=70?'var(--accent2)':t.p_calm_tomorrow>=55?'var(--warn)':'var(--danger)';
 const b75=t.bands.find(b=>b.target===75)||t.bands[0];
 return <div>
  <div style={{display:'flex',gap:12,flexWrap:'wrap'}}>
   <StatBox k="Tomorrow likely calm" v={t.p_calm_tomorrow+'%'} d={t.state==='calm'?'on a calm streak':'mixed / active now'} tone={calmTone}/>
   <StatBox k="Expected move (1 day)" v={'±'+t.sigma_pct+'%'} d="typical daily swing"/>
   <StatBox k="Vol regime" v={<Badge r={t.regime}/>} d={'stays same '+t.regime_persist_pct+'% of days'}/>
  </div>
  <div style={{marginTop:14}}><div className="sub" style={{marginBottom:6}}>Where tomorrow's close will most likely land — each row calibrated so it's right that % of the time on this market's own history</div>
   <table><thead><tr><th className="l">confidence</th><th>low</th><th>now</th><th>high</th><th>range</th></tr></thead>
    <tbody>{t.bands.map(b=><tr key={b.target}><td className="l"><b style={{color:b.target===75?'#00D4AA':'#8B83FF'}}>{b.target}%</b> chance inside</td>
      <td className="down">{fmt(b.low)}</td><td style={{color:'#6C63FF',fontWeight:700}}>{fmt(t.current_price)}</td>
      <td className="up">{fmt(b.high)}</td><td>{'±'+b.move_pct+'%'}</td></tr>)}</tbody></table></div>
 </div>;}
function Overview({d,sel,onPick}){
 const lc={calm:'#00D4AA',caution:'#FFB020',elevated:'#FF8C42',high:'#FF4757'};
 const rows=Object.keys(d).map(n=>{const x=d[n],po=x.posture||{},tm=x.tomorrow||{};
  return {n,score:po.score,level:po.level,regime:x.daily_regime,calm:tm.p_calm_tomorrow,move:tm.sigma_pct,opt:x.options};});
 return <table><thead><tr><th className="l">market</th><th>risk (0=calm)</th><th>regime</th><th>calm tomorrow</th><th>1-day move</th><th>options</th></tr></thead>
  <tbody>{rows.map(r=><tr key={r.n} onClick={()=>onPick(r.n)} style={{cursor:'pointer',background:r.n===sel?'rgba(108,99,255,.08)':'transparent'}}>
    <td className="l"><b>{r.n}</b></td>
    <td><span style={{fontFamily:'var(--mono)',fontWeight:700,color:lc[r.level]||'#8888AA'}}>{r.score!=null?r.score:'—'}</span> <span className="sub">{r.level||''}</span></td>
    <td><Badge r={r.regime}/></td>
    <td style={{color:r.calm>=78?'#00D4AA':r.calm>=60?'#FFB020':'#FF4757',fontWeight:600}}>{r.calm!=null?r.calm+'%':'—'}</td>
    <td>{r.move!=null?'±'+r.move+'%':'—'}</td>
    <td>{r.opt&&!r.opt.error?<span style={{color:r.opt.state==='cheap'?'#00D4AA':r.opt.state==='rich'?'#FF4757':'#FFB020',fontWeight:600}}>{r.opt.state}</span>:<span className="sub">—</span>}</td></tr>)}</tbody></table>;}
function ChangesPanel({c}){ if(!c||c.error) return <div className="sub">{(c&&c.error)||'n/a'}</div>;
 const sym={flip:'⇄',up:'▲',down:'▼',flat:'•'}; const col={flip:'#FFB020',up:'#FF4757',down:'#00D4AA',flat:'#8888AA'};
 return <div className="chips" style={{display:'flex',gap:10,flexWrap:'wrap'}}>
  {c.items.map((it,i)=><div key={i} style={{flex:'1 1 180px',minWidth:170,background:'var(--surface2)',border:'1px solid var(--line)',borderLeft:'3px solid '+(col[it.tone]||'#8888AA'),borderRadius:10,padding:'10px 13px'}}>
   <div className="sub" style={{fontSize:11,textTransform:'uppercase',letterSpacing:'.5px'}}>{it.label}</div>
   <div style={{marginTop:4,fontWeight:600,color:col[it.tone]||'var(--text)'}}>{sym[it.tone]||'•'} {it.detail}</div></div>)}
 </div>;}
function OptionsEdge({o}){ if(!o) return null; if(o.error) return <div className="sub">{o.error}</div>;
 const tone=o.state==='cheap'?'#00D4AA':o.state==='rich'?'#FF4757':'#FFB020';
 return <div>
  <div style={{display:'flex',gap:12,flexWrap:'wrap'}}>
   <StatBox k="Our vol forecast (HAR)" v={o.forecast_vol+'%'} d="expected realized vol"/>
   <StatBox k="Implied vol (India VIX)" v={o.implied_vol+'%'} d={'fear gauge · '+o.vix_pct+'th pct of history'}/>
   <StatBox k="Edge (forecast − implied)" v={(o.gap>=0?'+':'')+o.gap} d={(o.rel_pct>=0?'+':'')+o.rel_pct+'% vs implied'} tone={tone}/>
  </div>
  <div style={{marginTop:12,padding:'11px 14px',borderRadius:10,background:'rgba(108,99,255,.07)',borderLeft:'3px solid '+tone}}>
   <div style={{fontWeight:800,color:tone,textTransform:'uppercase',fontSize:12,letterSpacing:'.6px'}}>{o.state} · options</div>
   <div style={{marginTop:4,fontSize:14}}>{o.action}</div></div>
 </div>;}
function Field({label,val,set,step,suffix}){return <div style={{flex:'1 1 130px',minWidth:120}}>
 <div className="sub" style={{fontSize:11,textTransform:'uppercase',letterSpacing:'.5px',marginBottom:4}}>{label}</div>
 <input type="number" value={val} step={step||1} min={0} onChange={e=>set(Math.max(0,parseFloat(e.target.value)||0))}
  style={{width:'100%',background:'var(--surface2)',color:'var(--text)',border:'1px solid var(--line)',borderRadius:9,padding:'8px 10px',fontFamily:'var(--mono)',fontSize:14}}/>
 {suffix&&<div className="sub" style={{fontSize:10,marginTop:2}}>{suffix}</div>}</div>;}
function PositionSizer({t,regime}){ const [cap,setCap]=useState(100000),[risk,setRisk]=useState(1),[mult,setMult]=useState(1.5),[lot,setLot]=useState(1);
 if(!t||t.sigma_pct==null) return <div className="sub">n/a</div>;
 const price=t.current_price,sig=t.sigma_pct;
 const stopPts=mult*(sig/100)*price, stopPct=mult*sig;
 const maxRisk=cap*risk/100;
 const qty=stopPts>0?Math.floor(maxRisk/stopPts):0;
 const lots=lot>1?Math.floor(qty/lot):null;
 const exposure=qty*price, lev=cap>0?exposure/cap:0;
 const wild=regime==='explosive';
 const money=n=>'₹'+Math.round(n).toLocaleString('en-IN');
 const out=(k,v,d,tone)=><div style={{flex:'1 1 140px',minWidth:128,background:'var(--surface2)',border:'1px solid var(--line)',borderRadius:11,padding:'11px 13px'}}>
   <div className="sub" style={{fontSize:11,textTransform:'uppercase',letterSpacing:'.5px'}}>{k}</div>
   <div style={{fontFamily:'var(--mono)',fontSize:18,fontWeight:700,marginTop:5,color:tone||'var(--text)'}}>{v}</div>
   {d&&<div className="sub" style={{marginTop:2}}>{d}</div>}</div>;
 return <div>
  <div style={{display:'flex',gap:12,flexWrap:'wrap',marginBottom:14}}>
   <Field label="Capital (₹)" val={cap} set={setCap} step={10000}/>
   <Field label="Risk per trade %" val={risk} set={setRisk} step={0.25}/>
   <Field label="Stop width (× daily move)" val={mult} set={setMult} step={0.25} suffix={'1-day move ≈ ±'+sig+'%'}/>
   <Field label="Lot size" val={lot} set={setLot} step={1} suffix="1=shares · 75=NIFTY F&O…"/>
  </div>
  <div style={{display:'flex',gap:12,flexWrap:'wrap'}}>
   {out('Stop distance','±'+stopPct.toFixed(2)+'%',Math.round(stopPts).toLocaleString('en-IN')+' pts',wild?'#FFB020':undefined)}
   {out('Position size',qty.toLocaleString('en-IN')+(lots!=null?' · '+lots+' lots':''),'units to trade','#00D4AA')}
   {out('Exposure',money(exposure),lev.toFixed(1)+'× capital')}
   {out('Max loss if stopped',money(maxRisk),'= '+risk+'% of capital','#FF4757')}
   {out('Target (1:2 R:R)',Math.round(stopPts*2).toLocaleString('en-IN')+' pts','for 2× your risk')}
  </div>
  {wild&&<div className="sub" style={{marginTop:10,color:'#FFB020'}}>⚠ {regime} regime — swings are large now; consider a wider stop (2–2.5×) and smaller size.</div>}
 </div>;}
function OISpark({oi}){const ref=useRef();
 useEffect(()=>{if(!oi||!oi.spark||!window.Plotly)return;
  window.Plotly.react(ref.current,[{x:oi.spark.t,y:oi.spark.oi,mode:'lines',line:{color:'#6C63FF',width:1.6},fill:'tozeroy',fillcolor:'rgba(108,99,255,0.08)'}],
   {...PLOT,margin:{t:8,r:10,b:24,l:58},xaxis:{gridcolor:'#22222E',showticklabels:false},yaxis:{title:'total OI',gridcolor:'#22222E'}},{displayModeBar:false,responsive:true});
 },[oi]);return <div ref={ref} style={{height:170}}/>;}
function OIPanel({oi}){ if(!oi||oi.error) return <div className="sub">{(oi&&oi.error)||'no OI feed for this market'}</div>;
 const bias=oi.oi_net_bias, tone=bias==='bullish'?'var(--accent2)':bias==='bearish'?'var(--danger)':'var(--muted)';
 const bt=oi.oi_buildup, btT=bt==='long buildup'?'var(--accent2)':bt==='short buildup'?'var(--danger)':'var(--warn)';
 const cp=oi.oi_chg_pct_window, zz=Math.abs(oi.oi_chg_z);
 return <div>
  <div style={{display:'flex',gap:12,flexWrap:'wrap'}}>
   <StatBox k="Positioning now" v={bt} d={'net bias '+bias} tone={btT}/>
   <StatBox k="Net bias (recent)" v={bias} d="buildup vs unwinding" tone={tone}/>
   <StatBox k="Open interest" v={fmt(oi.oi_now)} d={(cp>=0?'+':'')+cp+'% over window'} tone={cp>=0?'var(--accent2)':'var(--danger)'}/>
   <StatBox k="OI flow (z-score)" v={oi.oi_chg_z} d={zz>1.5?'unusual flow':'normal flow'} tone={zz>1.5?'var(--warn)':undefined}/>
   <StatBox k="Bars with rising OI" v={oi.oi_rising_bars_pct+'%'} d="fresh positions building"/>
   <StatBox k="Rolled to later expiry" v={oi.rollover+'%'} d="near-month rollover"/>
  </div>
  <div style={{marginTop:14}}><div className="sub" style={{marginBottom:6}}>Total OI across expiries · recent (as of {oi.asof})</div><OISpark oi={oi}/></div>
  {oi.expiries&&oi.expiries.length>0&&<table style={{marginTop:12}}><thead><tr><th className="l">expiry</th><th>open interest</th><th>share</th></tr></thead>
   <tbody>{oi.expiries.map(e=><tr key={e.expiry}><td className="l">{e.expiry}</td><td>{fmt(e.oi)}</td><td>{e.share}%</td></tr>)}</tbody></table>}
 </div>;}
function MacroBanner({macro}){ if(!macro||macro.error) return null;
 const tone=macro.current==='risk-off'?'var(--danger)':macro.current==='risk-on'?'var(--accent2)':'var(--warn)';
 return <div className="panel" style={{display:'flex',alignItems:'center',gap:18,borderColor:tone,flexWrap:'wrap'}}>
  <div><div className="sub">Macro risk regime · Nifty + Crude + Gold</div>
   <div style={{fontSize:24,fontWeight:800,color:tone,marginTop:2}}>{macro.current.toUpperCase()}</div>
   <div className="sub">as of {macro.asof}</div></div>
  <div style={{flex:1,minWidth:20}}/>
  <table style={{width:'auto'}}><thead><tr><th className="l">regime</th><th>share</th><th>nifty</th><th>crude</th><th>gold</th></tr></thead>
   <tbody>{macro.rows.map(r=><tr key={r.regime}><td className="l">{r.regime}</td><td>{(r.share*100).toFixed(0)}%</td>
     <td className={r.nifty<0?'down':'up'}>{r.nifty}%</td><td>{r.crude}%</td><td className="up">{r.gold}%</td></tr>)}</tbody></table>
 </div>;}
function SwitchGauge({sw,regime}){ if(!sw||sw.error) return <div className="sub">{(sw&&sw.error)||'n/a'}</div>;
 const p=Math.round(sw.p_switch*100); const tone=p>=55?'var(--danger)':p>=35?'var(--warn)':'var(--accent2)';
 return <div>
  <div style={{display:'flex',alignItems:'flex-end',gap:12}}>
   <div style={{fontFamily:'var(--mono)',fontSize:34,fontWeight:800,color:tone,lineHeight:1}}>{p}<span className="sub" style={{fontSize:14}}>%</span></div>
   <div className="sub" style={{marginBottom:5}}>P(regime switch in next {sw.H} days)</div></div>
  <div className="gauge" style={{marginTop:9}}><div style={{width:p+'%'}}/></div>
  <div style={{display:'flex',gap:12,marginTop:14,flexWrap:'wrap'}}>
   <StatBox k="Model AUC" v={sw.auc!=null?sw.auc:'—'} d="switch reliability" tone={sw.auc>0.6?'var(--accent2)':undefined}/>
   <StatBox k="Change-point (BOCPD)" v={(sw.bocpd*100).toFixed(0)+'%'} d="distribution shift alarm" tone={sw.bocpd>0.3?'var(--warn)':undefined}/>
   <StatBox k="Now in regime" v={<Badge r={regime}/>} d={'as of '+sw.asof}/>
  </div></div>;}
function Plain({children}){return <div style={{background:'rgba(108,99,255,.07)',borderLeft:'3px solid #6C63FF',borderRadius:'0 8px 8px 0',padding:'9px 13px',margin:'10px 0 4px',fontSize:13.5,lineHeight:1.55,color:'#C2C2DA'}}><b style={{color:'#8B83FF'}}>In plain words — </b>{children}</div>;}
function RiskPosture({p,bandPct}){ if(!p||p.error) return <div className="sub">{(p&&p.error)||'n/a'}</div>;
 const C={calm:'#00D4AA',caution:'#FFB020',elevated:'#FF8C42',high:'#FF4757'}; const tone=C[p.level]||'#8888AA';
 return <div>
  <div style={{display:'flex',gap:24,flexWrap:'wrap',alignItems:'center'}}>
   <div style={{minWidth:150}}>
    <div style={{fontFamily:'var(--mono)',fontSize:46,fontWeight:800,color:tone,lineHeight:1}}>{p.score}<span className="sub" style={{fontSize:15}}>/100</span></div>
    <div style={{textTransform:'uppercase',letterSpacing:'1px',fontWeight:800,color:tone,marginTop:4,fontSize:15}}>{p.level}</div>
    <div className="gauge" style={{marginTop:10,width:150}}><div style={{width:p.score+'%',background:tone}}/></div>
    <div className="sub" style={{marginTop:7,fontSize:11,lineHeight:1.4}}><b style={{color:'#00D4AA'}}>0 = calm</b> · <b style={{color:'#FF4757'}}>100 = stormy</b><br/>lower is calmer — this is a risk gauge, not up/down</div>
   </div>
   <div style={{flex:1,minWidth:240}}>
    <div className="sub" style={{marginBottom:5}}>What to do</div>
    <div style={{fontSize:15,fontWeight:600,color:tone,marginBottom:10}}>{p.action}</div>
    {bandPct!=null&&<div className="sub">Expect a 20-day move up to <b style={{color:'#F0F0FF'}}>±{fmt(bandPct)}%</b> (auto-tuned 2σ)</div>}
   </div>
  </div>
  <div style={{marginTop:14}}><div className="sub" style={{marginBottom:6}}>Why — what's pushing the score right now</div>
   <div className="chips">{p.drivers.map((d,i)=><span key={i} className="badge" style={{background:d.push==='up'?'rgba(255,71,87,.15)':'rgba(0,212,170,.15)',color:d.push==='up'?'#FF4757':'#00D4AA'}}>{d.factor} {d.push==='up'?'↑ risk':'↓ risk'}</span>)}</div></div>
  <div style={{display:'flex',gap:12,marginTop:16,flexWrap:'wrap'}}>
   <StatBox k="Accuracy (AUC)" v={p.auc!=null?p.auc:'—'} d="spots high-vol periods" tone={p.auc>0.7?'#00D4AA':undefined}/>
   <StatBox k="Edge vs persistence" v={p.lift!=null?(p.lift>0?'+':'')+p.lift:'—'} d="what the factors add" tone={p.lift>0?'#00D4AA':'#FF4757'}/>
   <StatBox k="Best-case (GBM)" v={p.auc_gb!=null?p.auc_gb:'—'} d="nonlinear upper bound"/>
  </div>
  <details style={{marginTop:14}}><summary className="sub" style={{cursor:'pointer'}}>Factor weights — the scorecard (+ = raises risk, − = lowers)</summary>
   <table style={{marginTop:8}}><thead><tr><th className="l">factor</th><th>weight</th></tr></thead>
    <tbody>{p.weights.map((w,i)=><tr key={i}><td className="l">{w.factor}</td><td style={{color:w.coef>0?'#FF4757':'#00D4AA',fontWeight:600}}>{w.coef>0?'+':''}{w.coef}</td></tr>)}</tbody></table></details>
  </div>;}
function KPI({k,v,d,dir,i}){return <div className="kpi" style={{animationDelay:(i*80)+'ms'}}>
 <div className="k">{k}</div><div className="v">{v}</div>
 {d&&<div className={'d '+(dir||'flat')}>{dir==='up'?'▲':dir==='down'?'▼':'●'} {d}</div>}</div>;}

const NAV=[['#top','Overview',IC.dash],['#overview','All Markets',IC.grid],['#changes','What Changed',IC.chart],['#posture','Risk Decision',IC.shield],['#state','Market Mood',IC.shield],['#tomorrow','Tomorrow',IC.chart],['#options','Options Edge',IC.tag],['#sizer','Position Size',IC.tag],['#oi','Positioning',IC.layers],['#findings','What Works',IC.layers],['#cones','Price Swings',IC.chart],['#regime','Calm vs Wild',IC.layers],['#switch','Mood-Change Risk',IC.chart],['#price','Price Range',IC.tag],['#rel','Track Record',IC.shield],['#corr','What Moves Together',IC.grid]];
const FINDINGS=[
 ['Direction (intraday)','GBM / LSTM','AUC ~0.53 — efficient','b-explosive','near-efficient'],
 ['Movement / spike (30m)','GBM + HMM gate','AUC ~0.65 · top-5% → 69% vs 42%','b-normal','modest edge'],
 ['Daily realized vol','HAR-RV','R² 0.43–0.60 · beats GBM/transformers','b-quiet','strong'],
 ['Volatility regime','Gaussian HMM (causal)','~2× vol separation · 0.76–0.79 persist','b-quiet','real'],
 ['σ-band calibration','coverage backtest','indices ~68/87/95 · drift fixes trends','b-quiet','calibrated'],
];

function App(){
 const [d,setD]=useState(null),[corr,setCorr]=useState(null),[macro,setMacro]=useState(null),[ts,setTs]=useState(0),[sel,setSel]=useState(null),[act,setAct]=useState('#top');
 const apply=j=>{const inst=j.data.instruments||j.data;setD(inst);setCorr(j.data.correlation||null);setMacro(j.data.macro||null);setTs(j.ts);setSel(s=>s&&inst[s]?s:Object.keys(inst)[0]);};
 const load=()=>{if(window.__PRELOAD__){apply(window.__PRELOAD__);return;}fetch('/data').then(r=>r.json()).then(apply);};
 useEffect(()=>{load();if(window.__PRELOAD__)return;const t=setInterval(load,30000);return()=>clearInterval(t);},[]);
 if(!d||!sel)return <div style={{padding:40,color:'#8888AA'}}>Loading volatility intelligence…</div>;
 const x=d[sel],c20=x.forecast_cone.find(r=>r.H===20)||{},intr=x.intraday||{};
 const expand=c20.median-x.current; const dr=x.price_bands?x.price_bands.drift_daily_pct:0;
 const _b20=x.price_bands&&x.price_bands.bands_tuned?x.price_bands.bands_tuned.find(b=>b.H===20):null;
 const rangePct=_b20?(_b20.up2/x.price_bands.current_price-1)*100:null;
 return <div className="app">
  <aside className="sidebar">
   <div className="brand"><div className="logo">σ</div><span>VolIntel</span></div>
   <nav className="nav">{NAV.map(([h,l,ic])=><a key={h} href={h} className={act===h?'active':''} onClick={()=>setAct(h)}>{ic}<span>{l}</span></a>)}</nav>
   <div className="side-foot">HAR-RV · causal · walk-forward<br/>no look-ahead</div>
  </aside>
  <div className="main">
   <header className="header">
    <div><h1 id="top">Volatility Intelligence</h1><div className="sub">how much prices swing · the market's mood · what moves together · {Object.keys(d).length} markets</div></div>
    <div className="hgroup">
     <div className="live"><span className="pulse"></span>{window.__PRELOAD__?'snapshot':'live'}</div>
     <div className="iconbtn bell">{IC.bell}</div>
     <div className="avatar">RV</div>
    </div>
   </header>
   <div className="tabs">{Object.keys(d).map(n=><div key={n} className={'tab'+(n===sel?' on':'')} onClick={()=>setSel(n)}>{n}</div>)}</div>
   <div className="content">
    <div className="kpis">
     <KPI i={0} k={"Swings now · "+sel} v={fmt(x.current)+'%'} d={"how jumpy today · "+x.asof} dir="flat"/>
     <KPI i={1} k="Swings expected (next month)" v={fmt(c20.median)+'%'} d={(expand>=0?'+':'')+fmt(expand)+'% vs now · '+(expand>=0?'getting wilder':'calming down')} dir={expand>=0?'down':'up'}/>
     <KPI i={2} k="Market mood now" v={<Badge r={x.daily_regime}/>} d={"forecast fit R² "+fmt(c20.r2)} dir="flat"/>
     <KPI i={3} k="Stayed in range (past)" v={fmt(x.coverage['20']['raw']['2']*100)+'%'} d={'inside wide band · 1σ '+fmt(x.coverage['20']['raw']['1']*100)+'% · backtested'} dir={x.coverage['20']['raw']['2']>=0.9?'up':'down'}/>
    </div>

    <MacroBanner macro={macro}/>

    <div className="panel" id="overview"><h3>All Markets — At a Glance <span className="tag">{Object.keys(d).length} markets · click a row</span></h3>
     <Plain>Where to look first. Every market's risk score (0 = calm), current regime, tomorrow's calm-odds and expected move, and whether its options are cheap/rich — in one place. <b>Click any row</b> to open that market.</Plain>
     <Overview d={d} sel={sel} onPick={setSel}/></div>

    {x.changes&&!x.changes.error&&<div className="panel" id="changes"><h3>What Changed Today <span className="tag">{sel} · vs yesterday</span></h3>
     <Plain>The few things that actually moved since yesterday — so you don't re-read the whole board every morning. Regime flips, volatility jumps, the day's move, and where vol now sits in its yearly range.</Plain>
     <ChangesPanel c={x.changes}/></div>}

    <div className="panel" id="posture"><h3>Risk Decision Score <span className="tag">{sel} · multi-factor</span></h3>
     <Plain>The one-number summary: it blends every signal on this dashboard — vol forecast, regime, mood-change risk, macro, fat-tails — into a single 0–100 "how much risk is ahead" score, with a clear action. <b>It does not call up/down</b> (that's a coin-flip); it calls calm-vs-stormy, which is predictable. "Accuracy" = how well it spotted high-vol periods in the past; "Edge vs persistence" is the honest bit — how much the extra factors add beyond just "vol is already high."</Plain>
     <RiskPosture p={x.posture} bandPct={rangePct}/></div>

    <div className="panel" id="state"><h3>Market Mood — Health Check <span className="tag">{sel} · diagnostic</span></h3>
     <Plain>Is the market calm or stormy right now? Is the price trending in one direction or just bouncing around — and how bad could a rough day get? This is the weather report, not a buy/sell call.</Plain>
     <StatePanel s={x.state}/>
     <p className="sub" style={{marginTop:10}}>Diagnostics describe the regime (Hurst = trend vs mean-revert), tail risk (excess-kurtosis + EVT tail index), skew, and downside (VaR/CVaR, drawdown). State, not alpha.</p></div>

    {x.tomorrow&&<div className="panel" id="tomorrow"><h3>What Tomorrow Likely Looks Like <span className="tag">{sel} · next session</span></h3>
     <Plain>Honest next-day forecast. We <b>don't</b> predict up vs down — that's a coin-flip nobody beats. We predict <b>how the market behaves</b>: how big the swing will be, whether it stays calm, and the price range it'll most likely hold in. The <b style={{color:'#00D4AA'}}>75% row</b> is tuned so tomorrow's close lands inside it 3 times out of 4 — measured on this market's real history, not assumed.</Plain>
     <TomorrowPanel t={x.tomorrow}/>
     <p className="sub" style={{marginTop:10}}>Pure statistics on daily history — exploits volatility clustering (calm follows calm), which is genuinely predictable. Causal: σ = √(today's variance); the band multiplier is the empirical quantile of past |move|/σ. No direction bet, no look-ahead.</p></div>}

    {x.options&&<div className="panel" id="options"><h3>Options Edge — Cheap or Rich? <span className="tag">{sel} · vol risk premium</span></h3>
     <Plain>The one place this turns into a trade. We compare <b>our volatility forecast</b> (how much the market will actually move) against <b>implied volatility</b> (what option prices are charging — India VIX). Forecast higher → options are <b style={{color:'#00D4AA'}}>cheap, buy them</b>; forecast lower → options are <b style={{color:'#FF4757'}}>rich, sell them</b>. This monetizes the volatility edge — no up/down call needed.</Plain>
     <OptionsEdge o={x.options}/>
     <p className="sub" style={{marginTop:10}}>HAR realized-vol forecast (~20 trading days) vs India VIX (~30 calendar days). India VIX is NIFTY-only; BankNifty/commodities need their own listed option IV. Structural note: realized lands below implied ~60% of days (the variance risk premium), a mild base edge to selling.</p></div>}

    {x.tomorrow&&<div className="panel" id="sizer"><h3>Position Size &amp; Stops <span className="tag">{sel} · risk calculator</span></h3>
     <Plain>Turn the volatility forecast into an actual trade plan. Enter your capital and how much you're willing to risk — it sets a <b>stop distance</b> from the market's real daily move, and the <b>position size</b> so a stop-out costs exactly your chosen risk, never more. Vol-based, so stops auto-widen when the market turns wild.</Plain>
     <PositionSizer t={x.tomorrow} regime={x.daily_regime}/>
     <p className="sub" style={{marginTop:10}}>Stop = (your multiple) × today's 1-day move (σ). Size = max-₹-risk ÷ stop distance, so worst-case loss = capital × risk%. Educational sizing tool, not financial advice — set lot size for F&amp;O.</p></div>}

    {x.oi&&<div className="panel" id="oi"><h3>Futures Positioning — Open Interest <span className="tag">{sel} · who's in the trade</span></h3>
     <Plain>Open interest is how many futures contracts are live. Read alongside price it shows whether a move is backed by <b>fresh money</b> (strong) or just position-closing (weak): <b style={{color:'#00D4AA'}}>long buildup</b> = new buyers, <b style={{color:'#FF4757'}}>short buildup</b> = new sellers, <b>short covering</b> = a rally running out of fuel. Recent-data signal (~1 month of futures OI).</Plain>
     <OIPanel oi={x.oi}/>
     <p className="sub" style={{marginTop:10}}>Causal: OI forward-filled onto the price grid; buildup = sign(price move) × (rising OI ? strong : weak). Positioning context — blended into the direction call only when explicitly enabled.</p></div>}

    <div className="panel" id="findings"><h3>What We Tested — and What Actually Works <span className="tag">honest · backtested</span></h3>
     <Plain>What we tried, and what held up. Guessing tomorrow's up/down = basically a coin-flip (doesn't work). Predicting <i>how much</i> the price will swing = this works and is what the rest of the dashboard is built on.</Plain>
     <table><thead><tr><th className="l">signal</th><th className="l">method</th><th className="l">result</th><th className="l">verdict</th></tr></thead>
      <tbody>{FINDINGS.map((f,i)=><tr key={i}><td className="l">{f[0]}</td><td className="l sub">{f[1]}</td><td className="l">{f[2]}</td>
        <td className="l"><span className={'badge '+f[3]}>{f[4]}</span></td></tr>)}</tbody></table>
     <p className="sub" style={{marginTop:8}}>Honest result: intraday direction is a mirage; volatility forecasting is the real, defensible signal.</p></div>

    <div className="row2">
     <div className="panel" id="cones"><h3>Expected Price Swing <span className="tag">near · likely · extreme</span></h3>
      <Plain>How far the price could realistically move from here, going forward. A wider cone means bigger moves are expected ahead.</Plain><Cone inst={x}/></div>
     <div className="panel"><h3>Right Now (Today) <span className="tag">live · 30-min</span></h3>
      <Plain>Today, in real time: how close is a sudden sharp move (a "spike"), and which way it's leaning.</Plain>
      {intr.error||!intr.regime? <div className="sub">no intraday feed for this instrument</div>:
      <div><div style={{margin:'4px 0 14px'}}>mood now <Badge r={intr.regime}/></div>
       <div className="sub">chance of a sharp move soon</div><div className="gauge"><div style={{width:(intr.spike_pressure||0)+'%'}}/></div>
       <div style={{fontFamily:'var(--mono)',fontSize:24,fontWeight:700,marginTop:8}}>{intr.spike_pressure}<span className="sub" style={{fontSize:13}}>/100</span></div>
       <div style={{marginTop:10}} className="sub">bias: <b style={{color:intr.spike_dir==='up'?'#00D4AA':intr.spike_dir==='down'?'#FF4757':'#8888AA'}}>{intr.spike_dir}</b></div>
       <div className="sub" style={{marginTop:8}}>{intr.asof}</div></div>}
     </div>
    </div>

    <div className="row2" id="regime">
     <div className="panel"><h3>Expected Swing by Time-Frame <span className="tag">yearly %</span></h3>
      <Plain>Expected swing size over different windows — a few days, a week, a month out. Bigger number = bigger expected moves. The ±1σ / ±2σ columns are the likely vs extreme range.</Plain>
      <table><thead><tr><th className="l">H</th><th>-2σ</th><th>-1σ</th><th>median</th><th>+1σ</th><th>+2σ</th><th>R²</th></tr></thead>
       <tbody>{x.forecast_cone.map(r=><tr key={r.H}><td className="l">{r.H}d</td><td className="down">{fmt(r.dn2)}</td><td>{fmt(r.dn1)}</td>
         <td style={{color:'#6C63FF',fontWeight:700}}>{fmt(r.median)}</td><td>{fmt(r.up1)}</td><td className="up">{fmt(r.up2)}</td><td>{fmt(r.r2)}</td></tr>)}</tbody></table></div>
     <div className="panel"><h3>Range When Calm vs Wild <span className="tag">20d</span></h3>
      <Plain>If the market stays calm vs turns wild, here's the price range to expect. The "explosive" row is much wider — same market, very different risk.</Plain>
      <table><thead><tr><th className="l">regime</th><th>10%</th><th>median</th><th>90%</th><th>n</th></tr></thead>
       <tbody>{(x.regime_cones||[]).map(r=><tr key={r.regime}><td className="l"><Badge r={r.regime}/></td><td>{fmt(r.lo)}</td>
         <td style={{fontWeight:700}}>{fmt(r.median)}</td><td>{fmt(r.hi)}</td><td>{r.n}</td></tr>)}</tbody></table></div>
    </div>

    <div className="panel" id="switch"><h3>Mood-Change Early Warning <span className="tag">{sel} · next 10 days</span></h3>
     <Plain>The chance the market's mood flips (calm ↔ wild) within the next 10 days. A high % means expect a change soon. "Model AUC" is how often this warning was right in the past — closer to 1.0 = more trustworthy (0.5 would be a coin-flip). "Change-point" lights up the moment behaviour suddenly shifts.</Plain>
     <SwitchGauge sw={x.switch} regime={x.daily_regime}/>
     <p className="sub" style={{marginTop:12}}>Technical: P(regime switch ≤10d), HistGBM on vol-of-vol features, walk-forward AUC ~0.65–0.67. BOCPD = Bayesian change-point alarm. Predicts switches, not the exact next state.</p></div>

    <div className="panel" id="price"><h3>Where the Price Will Probably Stay <span className="tag">now: {fmt(x.price_bands&&x.price_bands.current_price)}</span>
      {x.price_bands&&x.price_bands.tuned&&<span className="badge" style={{marginLeft:8,background:'rgba(108,99,255,.18)',color:'#8B83FF'}}>AUTO-TUNED ×{fmt(x.price_bands.fatness)}</span>}</h3>
     <Plain>The price levels it'll most likely stay between. Roughly 2 days out of 3 it lands inside the 1σ band, ~19 of 20 inside 2σ. <b style={{color:'#8B83FF'}}>Use the green "Auto-tuned" table</b> — it widens the band by each market's real fat-tail history so the hit-ratio actually matches its promise. Left = textbook (no trend), middle = trend-tilted, right = auto-tuned (best).</Plain>
     <div style={{display:'flex',gap:16,flexWrap:'wrap'}}>
      <BandTable bands={x.price_bands.bands} price={x.price_bands.current_price} title="Textbook — no trend"/>
      <BandTable bands={x.price_bands.bands_drift} price={x.price_bands.current_price} title={'Trend-tilted — '+fmt(dr)+'%/day'}/>
      {x.price_bands.bands_tuned&&<BandTable bands={x.price_bands.bands_tuned} price={x.price_bands.current_price} title={'★ Auto-tuned (fat-tail ×'+fmt(x.price_bands.fatness)+')'}/>}</div></div>

    <div className="row2">
     <div className="panel" id="rel"><h3>Track Record — Were We Right? <span className="tag">{sel} · hit ratio</span></h3>
      <Plain>The report card — the hit ratio. Out of every past day, how often the real price actually stayed inside each band. Target is 68 / 87 / 95%; <b style={{color:'#00D4AA'}}>green = bang-on</b>, amber = a bit off, red = off. The <b style={{color:'#6C63FF'}}>"auto-tuned" row</b> is the calibrated band — it should sit closest to target.</Plain><Reliability cov={x.coverage}/>
      <p className="sub" style={{marginTop:8}}>Walk-forward coverage. "auto-tuned" learns each market's fat-tail multiplier on training folds only, then applies it to unseen test folds — honest, no look-ahead.</p></div>
     <div className="panel"><h3>How Jumpy It's Been <span className="tag">recent</span></h3>
      <Plain>The price's actual jumpiness recently, plotted day by day — the history behind the forecasts above.</Plain><Hist inst={x}/></div>
    </div>

    <div className="panel" id="corr"><h3>What Moves Together <span className="tag">daily</span></h3>
     <Plain>Which markets tend to move together (green) and which move opposite (red). Handy so you don't accidentally place the same bet twice — the equities all move together; Gold &amp; Silver track each other; commodities barely follow stocks.</Plain><Corr corr={corr}/>
     <p className="sub" style={{marginTop:8}}>Green = move together · red = inverse. Equity block clusters; Gold–Silver pair; commodities ≈ uncorrelated to equities.</p></div>

    <p className="sub" style={{marginTop:18}}>HAR-RV (Corsi 2009) on daily realized variance from 1-min bars · overnight-adjusted · updated {ts?new Date(ts*1000).toLocaleString():'snapshot'}. Research only.</p>
   </div>
  </div>
 </div>;
}
ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
</script>
<script>
(function(){
 var root=document.getElementById('root');
 function fail(t){root.innerHTML='<pre style="color:#FF4757;background:#111118;padding:28px;white-space:pre-wrap;font:13px monospace;line-height:1.5">DASHBOARD ERROR\n\n'+t+'</pre>';}
 try{
  if(!window.React||!window.ReactDOM) return fail('React CDN failed to load (check network/unpkg.com).');
  if(!window.Babel) return fail('Babel CDN failed to load (check network/unpkg.com).');
  if(!window.Plotly) return fail('Plotly CDN failed to load (check network/cdn.plot.ly).');
  var src=document.getElementById('appsrc').textContent;
  var code=Babel.transform(src,{presets:[['react',{runtime:'classic'}]]}).code;
  (0,eval)(code);
 }catch(e){ fail((e&&e.message||e)+'\n\n'+(e&&e.stack||'')); console.error(e); }
})();
</script>
</body></html>"""


if __name__ == "__main__":
    main()
