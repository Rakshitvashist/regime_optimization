"""
build_static_dashboard.py  —  static snapshot of the live dashboard for GitHub Pages.

GitHub Pages serves only static files, so we precompute all data and EMBED it in
the same React app (via window.__PRELOAD__). The result, docs/index.html, runs
fully client-side (React/Babel/Plotly from CDN) — no Python backend needed.

    python build_static_dashboard.py [--garch]
    -> docs/index.html   (commit + push; enable Pages: Settings > Pages > main /docs)
"""
from __future__ import annotations

import argparse, json, os, time

import vol_server
import vol_dashboard_data as vdd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--garch", action="store_true")
    a = ap.parse_args()
    print("Computing snapshot (first run ~1-2 min)...", flush=True)
    data = vdd.compute_all(garch=a.garch, log=lambda m: print(m, flush=True))
    payload = {"data": data, "ts": int(time.time())}
    tag = '<script type="text/plain" id="appsrc">'
    inject = "<script>window.__PRELOAD__ = " + json.dumps(payload) + ";</script>\n" + tag
    html = vol_server.PAGE.replace(tag, inject, 1)
    os.makedirs("docs", exist_ok=True)
    # Write to BOTH docs/ and repo root so GitHub Pages works whether its source
    # is "main /docs" OR "main /(root)" (a root index.html beats README rendering).
    for path in ("docs/index.html", "index.html"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
    open("docs/.nojekyll", "w").close()
    open(".nojekyll", "w").close()
    kb = os.path.getsize("index.html") / 1024
    print(f"\nWrote index.html + docs/index.html ({kb:.0f} KB, {len(data['instruments'])} instruments).")
    print("Commit + push. GitHub Pages works from main /(root) OR /docs.")


if __name__ == "__main__":
    main()
