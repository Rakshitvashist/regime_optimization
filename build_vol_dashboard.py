"""
build_vol_dashboard.py  —  generate a standalone volatility-cone web dashboard.

Computes, per instrument, the HAR realized-vol forecast + sigma bands (1s/1.5s/2s)
across horizons, plus the historical realized-vol series, and writes a single
self-contained vol_dashboard.html (Plotly via CDN). Serve it with:

    python -m http.server 8000      ->   http://localhost:8000/vol_dashboard.html

Sigma bands: HAR forecasts log(avg realized variance) over the next H days; the
out-of-sample residual std (sigma) gives the forecast uncertainty, so the band at
k-sigma is annvol(forecast +/- k*sigma). 1s~68%, 2s~95% coverage. All causal/walk-forward.
"""
from __future__ import annotations

import json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import daily_rv_forecast as drv
from vol_cone import daily_total_var, har_table, walk_forward

INSTRUMENTS = {"NIFTY 50": "NIFTY_50.csv", "BANK NIFTY": "BANKNIFTY.csv",
               "SENSEX": "SENSEX.csv", "RELIANCE": "RELIANCE.csv"}
HORIZONS = [5, 10, 20, 40, 60, 120]
SIGMAS = [1.0, 1.5, 2.0]


def compute(path, overnight=True):
    df = drv._load_1min(path)
    rv = daily_total_var(df, overnight=overnight)
    hist_vol = (np.sqrt(rv.rolling(20).mean() * drv.ANN) * 100).dropna()
    hist = {"date": [d.strftime("%Y-%m-%d") for d in hist_vol.index[::3]],
            "vol": [round(float(v), 2) for v in hist_vol.values[::3]]}
    cone = []
    for H in HORIZONS:
        F, target = har_table(rv, H)
        yt, yp, _ = walk_forward(F, target, H, 8, 0.4)
        sigma = float(np.std(yt - yp)); r2 = float(r2_score(yt, yp))
        comp = F.notna().all(axis=1) & target.notna()
        m = LinearRegression().fit(F[comp].to_numpy(), target[comp].to_numpy())
        last = F[F.notna().all(axis=1)].iloc[-1:]
        pt = float(m.predict(last.to_numpy())[0])
        e = {"H": H, "r2": round(r2, 3), "median": round(drv.annvol(pt), 2)}
        for k in SIGMAS:
            e[f"up{k}"] = round(drv.annvol(pt + k * sigma), 2)
            e[f"dn{k}"] = round(drv.annvol(pt - k * sigma), 2)
        cone.append(e)
    return {"hist": hist, "cone": cone, "asof": str(rv.index[-1].date()),
            "current": round(float(hist_vol.iloc[-1]), 2)}


HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>Volatility Cone Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#0e1117;color:#e6e6e6}
 header{padding:16px 24px;background:#161b22;border-bottom:1px solid #30363d}
 h1{margin:0;font-size:20px} .sub{color:#8b949e;font-size:13px;margin-top:4px}
 .wrap{padding:20px 24px;max-width:1100px;margin:auto}
 select{background:#21262d;color:#e6e6e6;border:1px solid #30363d;padding:6px 10px;border-radius:6px;font-size:14px}
 .cards{display:flex;gap:14px;margin:16px 0;flex-wrap:wrap}
 .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 16px;min-width:130px}
 .card .k{color:#8b949e;font-size:12px} .card .v{font-size:22px;font-weight:600;margin-top:4px}
 table{border-collapse:collapse;width:100%;margin-top:10px;font-size:13px}
 th,td{border:1px solid #30363d;padding:6px 10px;text-align:right} th{background:#21262d}
 td.l,th.l{text-align:left}
 #chart{height:430px} #hist{height:300px}
</style></head><body>
<header><h1>Volatility Cone Dashboard — Realized Vol Forecast (HAR-RV)</h1>
<div class="sub">Forward annualized volatility with 1&sigma; / 1.5&sigma; / 2&sigma; bands. Walk-forward, causal, leak-free.</div></header>
<div class="wrap">
 <label>Instrument: </label><select id="sel"></select>
 <span style="margin-left:16px;color:#8b949e" id="asof"></span>
 <div class="cards" id="cards"></div>
 <div id="chart"></div>
 <h3 style="margin-top:24px">Sigma bands by horizon (annualized vol %)</h3>
 <div id="tbl"></div>
 <h3 style="margin-top:24px">Historical realized volatility (20d, annualized)</h3>
 <div id="hist"></div>
 <p class="sub" style="margin-top:18px">HAR-RV (Corsi 2009) on daily realized variance built from 1-min bars (overnight-adjusted).
 Band = forecast &plusmn; k&times;&sigma;(out-of-sample residual). R&sup2; shown per horizon.</p>
</div>
<script>
const DATA = __DATA__;
const sel = document.getElementById('sel');
Object.keys(DATA).forEach(n=>{const o=document.createElement('option');o.value=n;o.text=n;sel.add(o);});
const SIG=[2.0,1.5,1.0], COL={2.0:'rgba(56,139,253,0.12)',1.5:'rgba(56,139,253,0.22)',1.0:'rgba(56,139,253,0.38)'};
function render(name){
 const d=DATA[name], c=d.cone, H=c.map(r=>r.H);
 document.getElementById('asof').textContent='as of '+d.asof+'  |  current 20d realized vol: '+d.current+'%';
 const cards=[['Current RV (20d)',d.current+'%'],['Forecast 20d (median)',c.find(r=>r.H==20).median+'%'],
   ['20d 1σ band',c.find(r=>r.H==20).dn1+' – '+c.find(r=>r.H==20).up1+'%'],
   ['20d 2σ band',c.find(r=>r.H==20).dn2+' – '+c.find(r=>r.H==20).up2+'%'],
   ['20d R²',c.find(r=>r.H==20).r2]];
 document.getElementById('cards').innerHTML=cards.map(x=>`<div class="card"><div class="k">${x[0]}</div><div class="v">${x[1]}</div></div>`).join('');
 let tr=[];
 SIG.forEach(k=>{
   tr.push({x:H,y:c.map(r=>r['up'+k]),mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'});
   tr.push({x:H,y:c.map(r=>r['dn'+k]),mode:'lines',line:{width:0},fill:'tonexty',fillcolor:COL[k],name:k+'σ',hoverinfo:'skip'});
 });
 tr.push({x:H,y:c.map(r=>r.median),mode:'lines+markers',line:{color:'#58a6ff',width:3},name:'median forecast'});
 Plotly.newPlot('chart',tr,{paper_bgcolor:'#0e1117',plot_bgcolor:'#0e1117',font:{color:'#e6e6e6'},
   margin:{t:10,r:10},xaxis:{title:'horizon (trading days ahead)',gridcolor:'#21262d'},
   yaxis:{title:'annualized vol %',gridcolor:'#21262d'},legend:{orientation:'h'}},{displayModeBar:false});
 // table
 let th='<table><tr><th class="l">Horizon</th><th>-2σ</th><th>-1.5σ</th><th>-1σ</th><th>median</th><th>+1σ</th><th>+1.5σ</th><th>+2σ</th><th>R²</th></tr>';
 c.forEach(r=>{th+=`<tr><td class="l">${r.H}d</td><td>${r.dn2}</td><td>${r['dn1.5']}</td><td>${r.dn1}</td><td><b>${r.median}</b></td><td>${r.up1}</td><td>${r['up1.5']}</td><td>${r.up2}</td><td>${r.r2}</td></tr>`;});
 document.getElementById('tbl').innerHTML=th+'</table>';
 Plotly.newPlot('hist',[{x:d.hist.date,y:d.hist.vol,mode:'lines',line:{color:'#3fb950',width:1.2}}],
   {paper_bgcolor:'#0e1117',plot_bgcolor:'#0e1117',font:{color:'#e6e6e6'},margin:{t:10,r:10},
    xaxis:{gridcolor:'#21262d'},yaxis:{title:'ann vol %',gridcolor:'#21262d'}},{displayModeBar:false});
}
sel.onchange=()=>render(sel.value); render(sel.value);
</script></body></html>"""


def main():
    data = {}
    for name, path in INSTRUMENTS.items():
        if os.path.exists(path):
            try:
                data[name] = compute(path)
                print(f"  {name:12s} -> ok (current {data[name]['current']}%)")
            except Exception as e:
                print(f"  {name:12s} -> FAILED: {type(e).__name__}: {e}")
    html = HTML.replace("__DATA__", json.dumps(data))
    with open("vol_dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nWrote vol_dashboard.html  ({len(data)} instruments)")
    print("Serve:  python -m http.server 8000   ->  http://localhost:8000/vol_dashboard.html")


if __name__ == "__main__":
    main()
