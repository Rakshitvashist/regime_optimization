"""
vol_server.py  —  live volatility dashboard (React frontend + Python backend).

Backend: stdlib http.server (no Flask). Computes the full dashboard data on
startup, caches it, and refreshes in a background thread every TTL. Serves:
    GET /        -> the React app (React 18 + Babel + Plotly, all via CDN)
    GET /data    -> cached JSON
    GET /health  -> status

Frontend: a real React app (components, hooks, auto-refresh) — no npm/build step
needed; React/Babel/Plotly load from CDN. Shows the forecast cone (1s/1.5s/2s),
the historical Burghardt cone overlay, optional GARCH line, per-regime cones,
and the live intraday HMM-regime + spike-imminence panel.

Run:
    set KMP_DUPLICATE_LIB_OK=TRUE   (handled below)
    python vol_server.py --port 8000 [--garch] [--ttl 900]
    -> open http://localhost:8000
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
    def log_message(self, *a):  # quiet
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
                                   "computing": _CACHE["computing"]}),
                       "application/json")
        elif self.path.startswith("/health"):
            self._send(json.dumps({"ok": True, "instruments": list(_CACHE["data"])}),
                       "application/json")
        else:
            self._send(PAGE)


def main():
    global _GARCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ttl", type=int, default=900, help="recompute interval (s)")
    ap.add_argument("--garch", action="store_true")
    a = ap.parse_args()
    _GARCH = a.garch
    print("Computing dashboard data (first run ~30-90s)...", flush=True)
    _recompute()
    threading.Thread(target=_bg_loop, args=(a.ttl,), daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    print(f"\nServing on http://localhost:{a.port}   (Ctrl+C to stop)", flush=True)
    srv.serve_forever()


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Volatility Cone — Live</title>
<script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 :root{--bg:#0e1117;--panel:#161b22;--line:#30363d;--mut:#8b949e;--acc:#58a6ff;--grn:#3fb950;--red:#f85149;--amb:#d29922}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:#e6e6e6;font-family:Segoe UI,Arial,sans-serif}
 header{padding:14px 22px;background:var(--panel);border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px}
 h1{margin:0;font-size:18px} .sub{color:var(--mut);font-size:12px}
 .wrap{max-width:1180px;margin:auto;padding:18px 22px}
 select{background:#21262d;color:#e6e6e6;border:1px solid var(--line);padding:6px 10px;border-radius:6px;font-size:14px}
 .row{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;flex:1;min-width:150px}
 .card .k{color:var(--mut);font-size:12px} .card .v{font-size:22px;font-weight:700;margin-top:5px}
 .badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:13px;font-weight:600}
 .b-quiet{background:rgba(63,185,80,.18);color:var(--grn)} .b-normal{background:rgba(210,153,34,.18);color:var(--amb)}
 .b-explosive{background:rgba(248,81,73,.18);color:var(--red)} .b-na{background:#21262d;color:var(--mut)}
 .panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}
 .panel h3{margin:0 0 10px;font-size:15px}
 table{border-collapse:collapse;width:100%;font-size:13px} th,td{border:1px solid var(--line);padding:6px 9px;text-align:right}
 th{background:#21262d} td.l,th.l{text-align:left}
 .gauge{height:10px;background:#21262d;border-radius:6px;overflow:hidden;margin-top:6px}
 .gauge>div{height:100%;background:linear-gradient(90deg,var(--grn),var(--amb),var(--red))}
 .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
</style></head><body><div id="root"></div>
<script type="text/babel">
const {useState,useEffect,useRef} = React;
const SIG=[2.0,1.5,1.0], FILL={2.0:'rgba(56,139,253,0.12)',1.5:'rgba(56,139,253,0.22)',1.0:'rgba(56,139,253,0.40)'};
const fmt=n=>n==null?'—':(typeof n==='number'?n.toFixed(n<100?2:0):n);

function Cone({inst}){
 const ref=useRef();
 useEffect(()=>{ if(!inst||!window.Plotly) return;
  const c=inst.forecast_cone, H=c.map(r=>r.H); let tr=[];
  SIG.forEach(k=>{
   tr.push({x:H,y:c.map(r=>r['up'+k]),mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'});
   tr.push({x:H,y:c.map(r=>r['dn'+k]),mode:'lines',line:{width:0},fill:'tonexty',fillcolor:FILL[k],name:k+'σ',hoverinfo:'skip'});
  });
  tr.push({x:H,y:c.map(r=>r.median),mode:'lines+markers',line:{color:'#58a6ff',width:3},name:'HAR median'});
  const hc=inst.hist_cone||[];
  if(hc.length){
   tr.push({x:hc.map(r=>r.H),y:hc.map(r=>r.median),mode:'lines',line:{color:'#8b949e',width:1.5,dash:'dot'},name:'hist median'});
   tr.push({x:hc.map(r=>r.H),y:hc.map(r=>r.p75),mode:'lines',line:{color:'#6e7681',width:1,dash:'dot'},name:'hist p75'});
   tr.push({x:hc.map(r=>r.H),y:hc.map(r=>r.p25),mode:'lines',line:{color:'#6e7681',width:1,dash:'dot'},name:'hist p25'});
  }
  if(inst.garch) tr.push({x:inst.garch.map(r=>r.H),y:inst.garch.map(r=>r.median),mode:'lines',line:{color:'#d29922',width:2,dash:'dash'},name:'GARCH'});
  window.Plotly.react(ref.current,tr,{paper_bgcolor:'#161b22',plot_bgcolor:'#161b22',font:{color:'#e6e6e6'},
   margin:{t:8,r:8,b:42,l:46},xaxis:{title:'horizon (days ahead)',gridcolor:'#21262d'},
   yaxis:{title:'annualized vol %',gridcolor:'#21262d'},legend:{orientation:'h',font:{size:11}}},{displayModeBar:false,responsive:true});
 },[inst]);
 return <div ref={ref} style={{height:430}}/>;
}
function Hist({inst}){
 const ref=useRef();
 useEffect(()=>{ if(!inst||!window.Plotly) return;
  window.Plotly.react(ref.current,[{x:inst.hist_series.date,y:inst.hist_series.vol,mode:'lines',line:{color:'#3fb950',width:1.1}}],
   {paper_bgcolor:'#161b22',plot_bgcolor:'#161b22',font:{color:'#e6e6e6'},margin:{t:8,r:8,b:30,l:46},
    xaxis:{gridcolor:'#21262d'},yaxis:{title:'ann vol %',gridcolor:'#21262d'}},{displayModeBar:false,responsive:true});
 },[inst]);
 return <div ref={ref} style={{height:260}}/>;
}
function Badge({r}){const cl=({quiet:'b-quiet',normal:'b-normal',explosive:'b-explosive'})[r]||'b-na';
 return <span className={'badge '+cl}>{r}</span>;}

function BandTable({bands,price,title}){
 return <div style={{flex:1,minWidth:300}}><div className="sub" style={{marginBottom:4}}>{title}</div>
  <table><thead><tr><th className="l">H</th><th>-2σ</th><th>-1.5σ</th><th>-1σ</th><th>price</th><th>+1σ</th><th>+1.5σ</th><th>+2σ</th></tr></thead>
   <tbody>{bands.map(b=><tr key={b.H}><td className="l">{b.H}d</td><td>{fmt(b.dn2)}</td><td>{fmt(b['dn1.5'])}</td><td>{fmt(b.dn1)}</td>
     <td><b>{fmt(price)}</b></td><td>{fmt(b.up1)}</td><td>{fmt(b['up1.5'])}</td><td>{fmt(b.up2)}</td></tr>)}</tbody></table></div>;}
function PriceRange({pb}){ if(!pb) return null;
 return <div style={{display:'flex',gap:14,flexWrap:'wrap'}}>
   <BandTable bands={pb.bands} price={pb.current_price} title="No-drift (pure volatility band)"/>
   <BandTable bands={pb.bands_drift} price={pb.current_price} title={'Drift-adjusted — recentered on trend ('+fmt(pb.drift_daily_pct)+'%/day)'}/>
  </div>;}

function Reliability({cov}){ if(!cov) return null;
 const rows=[]; Object.keys(cov).forEach(H=>['raw','drift'].forEach(mo=>rows.push([H,mo,cov[H][mo]])));
 const c=v=>({color:Math.abs(v-(v>0.8?(v>0.9?0.954:0.866):0.683))<0.04?'var(--grn)':'#e6e6e6'});
 return <div><table><thead><tr><th className="l">H</th><th className="l">band</th><th>1σ<br/><span className="sub">~68%</span></th>
    <th>1.5σ<br/><span className="sub">~87%</span></th><th>2σ<br/><span className="sub">~95%</span></th></tr></thead>
  <tbody>{rows.map(([H,mo,d],i)=><tr key={i}><td className="l">{mo==='raw'?H+'d':''}</td>
    <td className="l">{mo}</td><td>{(d['1']*100).toFixed(0)}%</td><td>{(d['1.5']*100).toFixed(0)}%</td><td>{(d['2']*100).toFixed(0)}%</td></tr>)}</tbody></table>
  <p className="sub" style={{marginTop:6}}>% of times realized price landed in the band. Close to 68/87/95 = well-calibrated; drift fixes trending names.</p></div>;}

function Corr({corr}){ const ref=useRef();
 useEffect(()=>{ if(!corr||!window.Plotly) return; const L=corr.labels;
  window.Plotly.react(ref.current,[{z:corr.matrix,x:L,y:L,type:'heatmap',zmin:-1,zmax:1,
    colorscale:[[0,'#f85149'],[0.5,'#0e1117'],[1,'#3fb950']],
    text:corr.matrix,texttemplate:'%{text}',textfont:{size:10},showscale:true}],
   {paper_bgcolor:'#161b22',plot_bgcolor:'#161b22',font:{color:'#e6e6e6'},margin:{t:8,r:8,b:90,l:90},
    xaxis:{tickangle:-40},yaxis:{autorange:'reversed'}},{displayModeBar:false,responsive:true});
 },[corr]);
 if(!corr) return null;
 return <div><div ref={ref} style={{height:380}}/>
  <div style={{marginTop:8}}><span className="sub">Strongest relationships: </span>
   {corr.top.map((p,i)=><span key={i} className="badge" style={{marginRight:6,
     background:p[2]>0?'rgba(63,185,80,.18)':'rgba(248,81,73,.18)',color:p[2]>0?'var(--grn)':'var(--red)'}}>{p[0]}–{p[1]} {p[2]}</span>)}
  </div></div>;}

function App(){
 const [d,setD]=useState(null),[corr,setCorr]=useState(null),[ts,setTs]=useState(0),[sel,setSel]=useState(null);
 const apply=(j)=>{const inst=j.data.instruments||j.data; setD(inst); setCorr(j.data.correlation||null);
   setTs(j.ts); setSel(s=>s&&inst[s]?s:Object.keys(inst)[0]);};
 const load=()=>{ if(window.__PRELOAD__){apply(window.__PRELOAD__);return;}
   fetch('/data').then(r=>r.json()).then(apply);};
 useEffect(()=>{load(); if(window.__PRELOAD__) return; const t=setInterval(load,30000);return()=>clearInterval(t);},[]);
 if(!d||!sel) return <div className="wrap">Loading dashboard…</div>;
 const inst=d[sel]; const c20=inst.forecast_cone.find(r=>r.H===20)||{};
 const intr=inst.intraday||{};
 return <div>
  <header>
   <div><h1>Volatility Cone — Live</h1>
    <div className="sub">HAR realized-vol forecast · 1σ/1.5σ/2σ bands · causal walk-forward · auto-refresh 30s</div></div>
   <div><select value={sel} onChange={e=>setSel(e.target.value)}>
     {Object.keys(d).map(n=><option key={n} value={n}>{n}</option>)}</select>
    <div className="sub" style={{marginTop:4}}>data as of {inst.asof} · updated {ts?new Date(ts*1000).toLocaleTimeString():'…'}</div></div>
  </header>
  <div className="wrap">
   <div className="row">
    <div className="card"><div className="k">Current RV (20d)</div><div className="v">{fmt(inst.current)}%</div></div>
    <div className="card"><div className="k">Forecast 20d (median)</div><div className="v">{fmt(c20.median)}%</div></div>
    <div className="card"><div className="k">20d 1σ band</div><div className="v" style={{fontSize:18}}>{fmt(c20.dn1)}–{fmt(c20.up1)}%</div></div>
    <div className="card"><div className="k">20d 2σ band</div><div className="v" style={{fontSize:18}}>{fmt(c20.dn2)}–{fmt(c20.up2)}%</div></div>
    <div className="card"><div className="k">Daily vol regime</div><div className="v" style={{fontSize:18}}><Badge r={inst.daily_regime}/></div></div>
   </div>

   <div className="row">
    <div className="panel" style={{flex:2,minWidth:520}}><h3>Forecast cone (σ bands) + historical cone {inst.garch?'+ GARCH':''}</h3><Cone inst={inst}/></div>
    <div className="panel" style={{flex:1,minWidth:260}}><h3>Live intraday signal (30m)</h3>
      {intr.error? <div className="sub">unavailable: {intr.error}</div> :
      <div>
       <div style={{margin:'6px 0'}}>HMM regime: <Badge r={intr.regime}/></div>
       <div className="sub" style={{marginTop:12}}>Spike-imminence pressure</div>
       <div className="gauge"><div style={{width:(intr.spike_pressure||0)+'%'}}/></div>
       <div style={{marginTop:6,fontSize:20,fontWeight:700}}>{fmt(intr.spike_pressure)}<span className="sub" style={{fontSize:13}}>/100</span></div>
       <div style={{marginTop:10}}><span className="dot" style={{background:intr.spike_dir==='up'?'var(--grn)':intr.spike_dir==='down'?'var(--red)':'var(--mut)'}}/>
        bias: {intr.spike_dir}</div>
       <div className="sub" style={{marginTop:10}}>as of {intr.asof}</div>
      </div>}
    </div>
   </div>

   <div className="row">
    <div className="panel" style={{flex:1,minWidth:300}}><h3>σ bands by horizon (ann vol %)</h3>
     <table><thead><tr><th className="l">H</th><th>-2σ</th><th>-1σ</th><th>median</th><th>+1σ</th><th>+2σ</th><th>R²</th></tr></thead>
      <tbody>{inst.forecast_cone.map(r=><tr key={r.H}><td className="l">{r.H}d</td><td>{fmt(r.dn2)}</td><td>{fmt(r.dn1)}</td>
        <td><b>{fmt(r.median)}</b></td><td>{fmt(r.up1)}</td><td>{fmt(r.up2)}</td><td>{fmt(r.r2)}</td></tr>)}</tbody></table>
    </div>
    <div className="panel" style={{flex:1,minWidth:300}}><h3>Regime cones (20d) — vol band by regime</h3>
     <table><thead><tr><th className="l">Regime</th><th>10%</th><th>median</th><th>90%</th><th>n</th></tr></thead>
      <tbody>{(inst.regime_cones||[]).map(r=><tr key={r.regime}><td className="l"><Badge r={r.regime}/></td>
        <td>{fmt(r.lo)}</td><td><b>{fmt(r.median)}</b></td><td>{fmt(r.hi)}</td><td>{r.n}</td></tr>)}</tbody></table>
    </div>
   </div>

   <div className="panel"><h3>Expected price range (σ) — {sel} @ {fmt(inst.price_bands&&inst.price_bands.current_price)}</h3>
    <PriceRange pb={inst.price_bands}/>
    <p className="sub" style={{marginTop:8}}>Price = current · exp(μ ± kσ), σ from the HAR vol forecast (7d / 20d); μ = recent drift.</p></div>

   <div className="panel" style={{maxWidth:560}}><h3>Band reliability (coverage backtest) — {sel}</h3>
    <Reliability cov={inst.coverage}/></div>

   <div className="panel"><h3>Cross-instrument correlation (daily returns) — find the relationships</h3>
    <Corr corr={corr}/>
    <p className="sub" style={{marginTop:8}}>Green = move together, red = inversely. Gold–Silver cluster, Crude–NatGas energy block, index block, etc.</p></div>

   <div className="panel"><h3>Historical realized volatility (20d, annualized)</h3><Hist inst={inst}/></div>
   <p className="sub">HAR-RV (Corsi 2009) on daily realized variance from 1-min bars (overnight-adjusted, matches India VIX).
    Bands = forecast ± kσ(out-of-sample residual). Regime = causal Gaussian-HMM. Spike pressure = percentile of current volume-z.</p>
  </div>
 </div>;
}
ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
</script></body></html>"""


if __name__ == "__main__":
    main()
