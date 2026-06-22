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
function Reliability({cov}){if(!cov)return null;const rows=[];Object.keys(cov).forEach(H=>['raw','drift'].forEach(m=>rows.push([H,m,cov[H][m]])));
 return <table><thead><tr><th className="l">H</th><th className="l">band</th><th>1σ <span className="sub">·68</span></th><th>1.5σ <span className="sub">·87</span></th><th>2σ <span className="sub">·95</span></th></tr></thead>
  <tbody>{rows.map(([H,m,d],i)=><tr key={i}><td className="l">{m==='raw'?H+'d':''}</td><td className="l" style={{color:m==='drift'?'#00D4AA':'#8888AA'}}>{m}</td>
   <td>{(d['1']*100).toFixed(0)}%</td><td>{(d['1.5']*100).toFixed(0)}%</td><td>{(d['2']*100).toFixed(0)}%</td></tr>)}</tbody></table>;}

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
function KPI({k,v,d,dir,i}){return <div className="kpi" style={{animationDelay:(i*80)+'ms'}}>
 <div className="k">{k}</div><div className="v">{v}</div>
 {d&&<div className={'d '+(dir||'flat')}>{dir==='up'?'▲':dir==='down'?'▼':'●'} {d}</div>}</div>;}

const NAV=[['#top','Overview',IC.dash],['#state','Market X-ray',IC.shield],['#findings','Findings',IC.layers],['#cones','Vol Cones',IC.chart],['#regime','Regimes',IC.layers],['#switch','Switch Risk',IC.chart],['#price','Price Bands',IC.tag],['#rel','Reliability',IC.shield],['#corr','Correlation',IC.grid]];
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
 return <div className="app">
  <aside className="sidebar">
   <div className="brand"><div className="logo">σ</div><span>VolIntel</span></div>
   <nav className="nav">{NAV.map(([h,l,ic])=><a key={h} href={h} className={act===h?'active':''} onClick={()=>setAct(h)}>{ic}<span>{l}</span></a>)}</nav>
   <div className="side-foot">HAR-RV · causal · walk-forward<br/>no look-ahead</div>
  </aside>
  <div className="main">
   <header className="header">
    <div><h1 id="top">Volatility Intelligence</h1><div className="sub">forecast · regimes · cross-asset · {Object.keys(d).length} instruments</div></div>
    <div className="hgroup">
     <div className="live"><span className="pulse"></span>{window.__PRELOAD__?'snapshot':'live'}</div>
     <div className="iconbtn bell">{IC.bell}</div>
     <div className="avatar">RV</div>
    </div>
   </header>
   <div className="tabs">{Object.keys(d).map(n=><div key={n} className={'tab'+(n===sel?' on':'')} onClick={()=>setSel(n)}>{n}</div>)}</div>
   <div className="content">
    <div className="kpis">
     <KPI i={0} k={"Current RV · "+sel} v={fmt(x.current)+'%'} d={"as of "+x.asof} dir="flat"/>
     <KPI i={1} k="Forecast 20d" v={fmt(c20.median)+'%'} d={(expand>=0?'+':'')+fmt(expand)+'% vs now · '+(expand>=0?'expanding':'contracting')} dir={expand>=0?'down':'up'}/>
     <KPI i={2} k="Daily Regime" v={<Badge r={x.daily_regime}/>} d={"R² "+fmt(c20.r2)} dir="flat"/>
     <KPI i={3} k="Price-in-Range 2σ" v={fmt(x.coverage['20']['raw']['2']*100)+'%'} d={'1σ '+fmt(x.coverage['20']['raw']['1']*100)+'% · 20d backtest'} dir={x.coverage['20']['raw']['2']>=0.9?'up':'down'}/>
    </div>

    <MacroBanner macro={macro}/>

    <div className="panel" id="state"><h3>Market State — X-ray <span className="tag">{sel} · diagnostic</span></h3>
     <StatePanel s={x.state}/>
     <p className="sub" style={{marginTop:10}}>Diagnostics describe the regime (Hurst = trend vs mean-revert), tail risk (excess-kurtosis + EVT tail index), skew, and downside (VaR/CVaR, drawdown). State, not alpha.</p></div>

    <div className="panel" id="findings"><h3>Model &amp; Research Findings <span className="tag">walk-forward · no look-ahead</span></h3>
     <table><thead><tr><th className="l">signal</th><th className="l">method</th><th className="l">result</th><th className="l">verdict</th></tr></thead>
      <tbody>{FINDINGS.map((f,i)=><tr key={i}><td className="l">{f[0]}</td><td className="l sub">{f[1]}</td><td className="l">{f[2]}</td>
        <td className="l"><span className={'badge '+f[3]}>{f[4]}</span></td></tr>)}</tbody></table>
     <p className="sub" style={{marginTop:8}}>Honest result: intraday direction is a mirage; volatility forecasting is the real, defensible signal.</p></div>

    <div className="row2">
     <div className="panel" id="cones"><h3>Forecast Cone <span className="tag">1σ · 1.5σ · 2σ</span></h3><Cone inst={x}/></div>
     <div className="panel"><h3>Live Intraday <span className="tag">30m HMM</span></h3>
      {intr.error||!intr.regime? <div className="sub">no intraday feed for this instrument</div>:
      <div><div style={{margin:'4px 0 14px'}}>regime <Badge r={intr.regime}/></div>
       <div className="sub">spike-imminence</div><div className="gauge"><div style={{width:(intr.spike_pressure||0)+'%'}}/></div>
       <div style={{fontFamily:'var(--mono)',fontSize:24,fontWeight:700,marginTop:8}}>{intr.spike_pressure}<span className="sub" style={{fontSize:13}}>/100</span></div>
       <div style={{marginTop:10}} className="sub">bias: <b style={{color:intr.spike_dir==='up'?'#00D4AA':intr.spike_dir==='down'?'#FF4757':'#8888AA'}}>{intr.spike_dir}</b></div>
       <div className="sub" style={{marginTop:8}}>{intr.asof}</div></div>}
     </div>
    </div>

    <div className="row2" id="regime">
     <div className="panel"><h3>σ Bands by Horizon <span className="tag">ann vol %</span></h3>
      <table><thead><tr><th className="l">H</th><th>-2σ</th><th>-1σ</th><th>median</th><th>+1σ</th><th>+2σ</th><th>R²</th></tr></thead>
       <tbody>{x.forecast_cone.map(r=><tr key={r.H}><td className="l">{r.H}d</td><td className="down">{fmt(r.dn2)}</td><td>{fmt(r.dn1)}</td>
         <td style={{color:'#6C63FF',fontWeight:700}}>{fmt(r.median)}</td><td>{fmt(r.up1)}</td><td className="up">{fmt(r.up2)}</td><td>{fmt(r.r2)}</td></tr>)}</tbody></table></div>
     <div className="panel"><h3>Regime Cones <span className="tag">20d</span></h3>
      <table><thead><tr><th className="l">regime</th><th>10%</th><th>median</th><th>90%</th><th>n</th></tr></thead>
       <tbody>{(x.regime_cones||[]).map(r=><tr key={r.regime}><td className="l"><Badge r={r.regime}/></td><td>{fmt(r.lo)}</td>
         <td style={{fontWeight:700}}>{fmt(r.median)}</td><td>{fmt(r.hi)}</td><td>{r.n}</td></tr>)}</tbody></table></div>
    </div>

    <div className="panel" id="switch"><h3>Regime-Switch Early-Warning <span className="tag">{sel} · P(switch ≤10d)</span></h3>
     <SwitchGauge sw={x.switch} regime={x.daily_regime}/>
     <p className="sub" style={{marginTop:12}}>Predicts a regime change in the next 10 days (walk-forward AUC ~0.65–0.67, driven by vol-of-vol). BOCPD = live change-point alarm. This is the predictable part of regime — switches, not the exact next state.</p></div>

    <div className="panel" id="price"><h3>Expected Price Range <span className="tag">{fmt(x.price_bands&&x.price_bands.current_price)}</span></h3>
     <div style={{display:'flex',gap:16,flexWrap:'wrap'}}>
      <BandTable bands={x.price_bands.bands} price={x.price_bands.current_price} title="No-drift — pure volatility band"/>
      <BandTable bands={x.price_bands.bands_drift} price={x.price_bands.current_price} title={'Drift-adjusted — trend '+fmt(dr)+'%/day'}/></div></div>

    <div className="row2">
     <div className="panel" id="rel"><h3>Price-in-Range — Band Hit-Rate <span className="tag">{sel}</span></h3><Reliability cov={x.coverage}/>
      <p className="sub" style={{marginTop:8}}>How often the price actually landed inside the band (backtested). Near 68/87/95% = calibrated; drift fixes trending names.</p></div>
     <div className="panel"><h3>Historical Realized Vol <span className="tag">20d</span></h3><Hist inst={x}/></div>
    </div>

    <div className="panel" id="corr"><h3>Cross-Asset Correlation <span className="tag">daily returns</span></h3><Corr corr={corr}/>
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
