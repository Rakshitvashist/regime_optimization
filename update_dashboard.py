"""
update_dashboard.py  —  One-command full pipeline
==================================================
Runs the complete Nifty 50 + Nifty 500 dashboard update and deploys to GitHub.

Usage:
    python update_dashboard.py

Flags (all optional):
    --skip-download       Skip yfinance downloads (use cached CSVs)
    --skip-indicators     Skip indicator computation for Nifty 50
    --skip-nifty50        Skip Nifty 50 pipeline entirely
    --skip-nifty500       Skip Nifty 500 pipeline entirely
    --skip-push           Skip git commit/push (local preview only)
    --no-color            Disable coloured output
    --optimize            Learn tuned indicator/weight configs (feature 001) before
                          generating summaries; off by default
"""

import subprocess, sys, os, time, argparse
from datetime import datetime

# ── ANSI colours (auto-disable on Windows if no ANSI support) ────────────────
RESET = GREEN = YELLOW = RED = CYAN = BOLD = ""
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)
    RESET  = "\033[0m";  GREEN  = "\033[92m";  YELLOW = "\033[93m"
    RED    = "\033[91m"; CYAN   = "\033[96m";  BOLD   = "\033[1m"
except Exception:
    pass

def log(msg, color=""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{ts}] {msg}{RESET}", flush=True)

def banner(title):
    line = "=" * 60
    print(f"\n{CYAN}{BOLD}{line}\n  {title}\n{line}{RESET}\n", flush=True)

def run(cmd, label):
    """Run a subprocess command; raise on failure."""
    log(f"▶ {label}", CYAN)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable] + cmd if cmd[0].endswith(".py") else cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=False   # let output stream live to terminal
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        log(f"✗ FAILED: {label}  (exit {result.returncode})  [{elapsed:.0f}s]", RED)
        raise SystemExit(result.returncode)
    log(f"✓ Done: {label}  [{elapsed:.0f}s]", GREEN)

def git_commit_push(message):
    """Add changed public JSON files, commit, and push."""
    base = os.path.dirname(os.path.abspath(__file__))

    def git(args, check=True):
        r = subprocess.run(["git"] + args, cwd=base, capture_output=True, text=True)
        if check and r.returncode != 0:
            log(f"git {args[0]} failed: {r.stderr.strip()}", RED)
            raise SystemExit(r.returncode)
        return r.stdout.strip()

    # Stage only the JSON outputs (avoid staging huge raw CSVs)
    git(["add", "dashboard/public/summary.json",
                 "dashboard/public/summary_500.json"])

    status = git(["status", "--porcelain"], check=False)
    if not status:
        log("Nothing new to commit — dashboard JSON is already up to date.", YELLOW)
        return

    git(["commit", "-m", message])
    log("Pushing to GitHub …", CYAN)
    git(["push", "origin", "main"])
    log("✓ Pushed! GitHub Actions will redeploy in ~2-3 minutes.", GREEN)


def main():
    parser = argparse.ArgumentParser(description="Full Nifty 50/500 dashboard pipeline")
    parser.add_argument("--skip-download",   action="store_true", help="Skip yfinance download step")
    parser.add_argument("--skip-indicators", action="store_true", help="Skip 400-indicator computation for Nifty 50")
    parser.add_argument("--skip-nifty50",    action="store_true", help="Skip entire Nifty 50 pipeline")
    parser.add_argument("--skip-nifty500",   action="store_true", help="Skip entire Nifty 500 pipeline")
    parser.add_argument("--skip-push",       action="store_true", help="Skip git commit/push")
    parser.add_argument("--no-color",        action="store_true", help="Disable colour output")
    parser.add_argument("--optimize",        action="store_true",
                        help="Run indicator/weight optimization (feature 001) before summaries")
    args = parser.parse_args()

    overall_start = time.time()

    print(f"""
{CYAN}{BOLD}
╔══════════════════════════════════════════════════════════╗
║       NIFTY DASHBOARD — FULL UPDATE PIPELINE             ║
║  Nifty 50 + Nifty 500  →  JSON  →  GitHub Pages         ║
╚══════════════════════════════════════════════════════════╝
{RESET}""")

    # ── NIFTY 50 ──────────────────────────────────────────────────────────────
    if not args.skip_nifty50:
        banner("STEP 1 · Nifty 50 — Download (incremental)")
        if not args.skip_download:
            run(["nifty_50.py"], "Download Nifty 50 OHLCV (yfinance incremental)")
        else:
            log("Skipping download (--skip-download)", YELLOW)

        banner("STEP 2 · Nifty 50 — Compute 400+ Indicators")
        if not args.skip_indicators:
            run(["start.py",
                 "--input",  "nifty50_host",
                 "--output", "processed_indicators"],
                "Compute 400+ indicators for Nifty 50")
        else:
            log("Skipping indicators (--skip-indicators)", YELLOW)

        if args.optimize:
            banner("STEP 2.5 · Nifty 50 — Optimize indicators + weights (feature 001)")
            run(["optimize_indicators.py", "--scope", "batch",
                 "--input", "processed_indicators", "--configs-dir", "configs"],
                "Optimize tuned configs (Nifty 50)")

        banner("STEP 3 · Nifty 50 — Generate Dashboard JSON")
        run(["generate_summary.py",
             "--input",  "processed_indicators",
             "--output", "dashboard/public/summary.json"],
            "Generate summary.json (Nifty 50)")
    else:
        log("Skipping Nifty 50 pipeline (--skip-nifty50)", YELLOW)

    # ── NIFTY 500 ─────────────────────────────────────────────────────────────
    if not args.skip_nifty500:
        banner("STEP 4 · Nifty 500 — Download (incremental)")
        if not args.skip_download:
            run(["nifty_500.py"], "Download Nifty 500 OHLCV (yfinance incremental)")
        else:
            log("Skipping download (--skip-download)", YELLOW)

        banner("STEP 4.5 · Nifty 500 — Compute 400+ Indicators")
        if not args.skip_indicators:
            run(["start.py",
                 "--input",  "nifty500_host",
                 "--output", "processed_indicators_500"],
                "Compute 400+ indicators for Nifty 500")
        else:
            log("Skipping indicators (--skip-indicators)", YELLOW)

        banner("STEP 5 · Nifty 500 — Generate Dashboard JSON")
        run(["generate_summary.py",
             "--input",  "processed_indicators_500",
             "--output", "dashboard/public/summary_500.json"],
            "Generate summary_500.json (Nifty 500, all 500 stocks)")
    else:
        log("Skipping Nifty 500 pipeline (--skip-nifty500)", YELLOW)

    # ── GIT PUSH ──────────────────────────────────────────────────────────────
    if not args.skip_push:
        banner("STEP 6 · Commit & Push to GitHub")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        git_commit_push(f"Auto-update dashboard data [{ts}]")
    else:
        log("Skipping git push (--skip-push)", YELLOW)

    elapsed = time.time() - overall_start
    print(f"""
{GREEN}{BOLD}
╔══════════════════════════════════════════════════════════╗
║  ✓  ALL STEPS COMPLETE  in {int(elapsed//60)}m {int(elapsed%60):02d}s
║  Dashboard live in ~2-3 min at GitHub Pages              ║
╚══════════════════════════════════════════════════════════╝
{RESET}""")


if __name__ == "__main__":
    main()
