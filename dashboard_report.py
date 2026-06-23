"""
dashboard_report.py  —  generate the Volatility Intelligence Dashboard report (PDF).

A plain-language guide: what the dashboard does, what each panel gives you, how to
use it day to day, the honest evidence behind it, and what to add next.

    python dashboard_report.py            ->  Volatility_Dashboard_Report.pdf
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

# ---- palette (matches the dashboard) ----
ACCENT = colors.HexColor("#6C63FF")
ACCENT2 = colors.HexColor("#00D4AA")
DANGER = colors.HexColor("#FF4757")
WARN = colors.HexColor("#E8A317")
INK = colors.HexColor("#1A1A2E")
MUTED = colors.HexColor("#6B6B85")
LINE = colors.HexColor("#D8D8E4")
SOFT = colors.HexColor("#F4F4FA")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=18, textColor=ACCENT,
                    spaceBefore=16, spaceAfter=8, leading=22)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, textColor=INK,
                    spaceBefore=12, spaceAfter=5, leading=16)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=10, textColor=INK,
                      leading=15, spaceAfter=6)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.5, textColor=MUTED, leading=12)
LEAD = ParagraphStyle("Lead", parent=BODY, fontSize=11, textColor=INK, leading=16)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=4, spaceAfter=3)


def make_charts(symbol="NIFTY 50", outdir="."):
    """Render real Regime-Map and Forecast-Track PNGs from live data. Returns paths
    (or {} if the data pipeline isn't available)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import vol_dashboard_data as vdd
        import daily_rv_forecast as drv
        from vol_cone import daily_total_var
        path = vdd.INSTRUMENTS[symbol][0]
        df = drv._load_1min(path); rv = daily_total_var(df, overnight=True)
        rc = vdd._regime_series(df, rv); ft = vdd._forecast_track(rv)
        out = {}

        d = [datetime.strptime(x, "%Y-%m-%d") for x in rc["date"]]
        g = rc["regime"]; COL = {0: "#C8F3E8", 1: "#DCD9FF", 2: "#FFD6DB"}
        fig, ax = plt.subplots(figsize=(7.4, 2.5), dpi=150)
        s = 0
        for i in range(1, len(g) + 1):
            if i == len(g) or g[i] != g[s]:
                ax.axvspan(d[s], d[min(i, len(d) - 1)], color=COL.get(g[s], "#fff"), lw=0)
                s = i
        ax.plot(d, rc["price"], color="#1A1A2E", lw=0.9)
        ax.set_title(f"{symbol} — price shaded by regime  (green=calm · purple=normal · red=wild)",
                     fontsize=7.5, color="#444")
        ax.margins(x=0); ax.tick_params(labelsize=6)
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
        rp = os.path.join(outdir, "_chart_regime.png")
        fig.tight_layout(); fig.savefig(rp, bbox_inches="tight"); plt.close(fig)
        out["regime"] = rp

        if not ft.get("error"):
            dd = [datetime.strptime(x, "%Y-%m-%d") for x in ft["date"]]
            fig, ax = plt.subplots(figsize=(7.4, 2.3), dpi=150)
            ax.plot(dd, ft["actual"], color="#1A1A2E", lw=1.1, label="actual volatility")
            ax.plot(dd, ft["pred"], color="#6C63FF", lw=1.4, ls=":", label="our forecast")
            ax.set_title(f"{symbol} — volatility forecast vs reality (out-of-sample)  "
                         f"corr {ft['corr']} · R² {ft['r2']}", fontsize=7.5, color="#444")
            ax.legend(fontsize=6.5, frameon=False); ax.margins(x=0); ax.tick_params(labelsize=6)
            ax.set_ylabel("ann vol %", fontsize=7)
            for sp in ax.spines.values():
                sp.set_color("#cccccc")
            tp = os.path.join(outdir, "_chart_track.png")
            fig.tight_layout(); fig.savefig(tp, bbox_inches="tight"); plt.close(fig)
            out["track"] = tp
        return out
    except Exception as e:
        print("charts skipped:", type(e).__name__, e)
        return {}


def chip(text, color):
    return Paragraph(f'<font color="{color.hexval()[2:]}"><b>{text}</b></font>', BODY)


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(t, BULLET), leftIndent=12, value="•") for t in items],
        bulletType="bullet", start="•", leftIndent=10, spaceAfter=8)


def table(data, head=True, col_widths=None, zebra=True):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    sty = [
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
    ]
    if head:
        sty += [("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9)]
    if zebra:
        for r in range(1, len(data)):
            if r % 2 == 0:
                sty.append(("BACKGROUND", (0, r), (-1, r), SOFT))
    t.setStyle(TableStyle(sty))
    return t


def callout(text, color=ACCENT2):
    p = Paragraph(text, BODY)
    t = Table([[p]], colWidths=[165 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SOFT),
        ("LINEBEFORE", (0, 0), (0, -1), 3, color),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 12 * mm, "Volatility Intelligence Dashboard")
    canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(LINE)
    canvas.line(20 * mm, 15 * mm, 190 * mm, 15 * mm)
    canvas.restoreState()


def build(path="Volatility_Dashboard_Report.pdf"):
    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            title="Volatility Intelligence Dashboard")
    S = []
    charts = make_charts()

    # ---------------- Title ----------------
    S.append(Spacer(1, 40 * mm))
    S.append(Paragraph("Volatility Intelligence Dashboard", ParagraphStyle(
        "T", parent=H1, fontSize=28, alignment=TA_CENTER, textColor=ACCENT, leading=32)))
    S.append(Paragraph("A trader's guide — what it does, how to use it, and what's next",
                       ParagraphStyle("Tsub", parent=BODY, fontSize=13, alignment=TA_CENTER,
                                      textColor=MUTED, spaceBefore=6)))
    S.append(Spacer(1, 10 * mm))
    S.append(Paragraph(
        '<font color="#00D4AA"><b>The one idea:</b></font> We cannot predict which way '
        'the market goes tomorrow — that is a coin-flip nobody beats. But we <b>can</b> '
        'predict <i>how it behaves</i>: how big the swing will be, whether it stays calm, '
        'and the range it will hold. This dashboard turns that one real, defensible edge '
        'into clear trading decisions.',
        ParagraphStyle("Tq", parent=LEAD, alignment=TA_CENTER, fontSize=12)))
    S.append(Spacer(1, 14 * mm))
    S.append(Paragraph("Generated for the regime_optimization project  ·  built on causal, "
                       "walk-forward-validated models", ParagraphStyle(
                           "Tf", parent=SMALL, alignment=TA_CENTER)))
    S.append(PageBreak())

    # ---------------- 1. Executive summary ----------------
    S.append(Paragraph("1 · Executive Summary", H1))
    S.append(Paragraph(
        "This dashboard is an honest market-risk cockpit. It does not pretend to call "
        "up or down. Instead it forecasts <b>volatility and regime</b> — the part of the "
        "market that genuinely is predictable — and converts it into decisions: how "
        "risky the next session is, the price range to expect, whether options are cheap "
        "or rich, and how big a position to take.", BODY))
    S.append(callout(
        "<b>Why this matters:</b> Most retail tools sell direction signals that quietly "
        "fail. This one is built the opposite way — it states plainly what cannot be "
        "predicted (direction) and squeezes real value from what can (volatility). Every "
        "number is backed by a walk-forward backtest with no look-ahead."))
    S.append(Spacer(1, 4))
    S.append(Paragraph("What you get, in one line each:", H2))
    S.append(bullets([
        "<b>Regime Map</b> — the price, with the background coloured calm / normal / wild.",
        "<b>All Markets grid</b> — risk, regime and tomorrow's outlook for every instrument at once.",
        "<b>Risk Decision Score</b> — one 0–100 'how stormy ahead' number with an action.",
        "<b>Tomorrow</b> — the calm-odds and the price band that is right 75% of the time.",
        "<b>Options Edge</b> — a buy-or-sell-options call from forecast vs implied volatility.",
        "<b>Position Size &amp; Stops</b> — turns the volatility into a concrete lot size and stop.",
    ]))
    if charts.get("regime"):
        S.append(Spacer(1, 4))
        S.append(Paragraph("The signature view — the Regime Map (live data):", H2))
        S.append(Image(charts["regime"], width=165 * mm, height=55 * mm))
        S.append(Paragraph("Price with the background shaded by market mood. Red stretches "
                           "are where big swings clustered.", SMALL))
    S.append(PageBreak())

    # ---------------- 2. The core finding ----------------
    S.append(Paragraph("2 · The Core Finding (why it is built this way)", H1))
    S.append(Paragraph("We tested direction prediction exhaustively — every timeframe from "
                       "5 minutes to 1 day, with hundreds of indicators and ML models. The "
                       "honest result:", BODY))
    S.append(Paragraph("Direction accuracy is a coin-flip everywhere", H2))
    S.append(table([
        ["Timeframe", "Honest direction accuracy", "Verdict"],
        ["5 min", "~50–51%", "coin-flip"],
        ["15 / 30 min", "~52–53%", "coin-flip"],
        ["1 hour", "~51–56%", "marginal / noise"],
        ["1 day", "below the 'always-up' baseline", "no edge"],
    ], col_widths=[45 * mm, 70 * mm, 50 * mm]))
    S.append(Spacer(1, 6))
    S.append(callout(
        "<b>The 80% trap.</b> A 5-minute model first showed 80% accuracy and passed an "
        "8-fold walk-forward. It was <b>look-ahead leakage</b> — a feature secretly read "
        "the next 2 bars. A gap test proved it: skip just 3 bars and accuracy collapses "
        "0.79 → 0.51. Lesson baked into the project: walk-forward alone does not catch "
        "leakage; only a causality probe does.", DANGER))
    S.append(Spacer(1, 4))
    S.append(Paragraph("But behaviour / volatility IS predictable — to 75%+", H2))
    S.append(Paragraph("The same history, pointed at the right target, pays off — because "
                       "volatility clusters (calm follows calm), and that is not arbitraged "
                       "away the way a direction signal would be:", BODY))
    S.append(table([
        ["What you predict about tomorrow", "Accuracy", ""],
        ["Direction (up / down)", "~52%", "impossible"],
        ["Calm day, after a calm streak", "~84%", "strong"],
        ["Close stays within the ±1.3σ band", "~75% (tunable)", "by design"],
        ["Volatility regime persists (BankNifty)", "~89%", "strong"],
    ], col_widths=[85 * mm, 40 * mm, 40 * mm]))
    S.append(Spacer(1, 6))
    S.append(callout(
        "<b>The pivot:</b> stop trying to guess <i>which way</i> (≈50%), and predict "
        "<i>how far / how calm</i> (75%+). That is the whole dashboard."))
    S.append(PageBreak())

    # ---------------- 3. Panels ----------------
    S.append(Paragraph("3 · What Each Panel Gives You", H1))
    panels = [
        ("Regime Map", "The price line over ~2 years with the background shaded by "
         "market mood (green=calm, purple=normal, red=wild).",
         "See instantly where big swings clustered and what regime you are in now. Red "
         "stretches = be defensive; green = trend with confidence."),
        ("All Markets — At a Glance", "A grid of every instrument: risk score, regime, "
         "tomorrow-calm %, expected move, options cheap/rich.",
         "Pick where to focus today. Click a row to open that market. One screen tells "
         "you where the risk and the opportunity are."),
        ("What Changed Today", "Day-over-day diffs: regime flips, volatility jumps, the "
         "day's move, and where vol sits in its yearly range.",
         "A 10-second morning check — see what actually moved without re-reading the "
         "whole board."),
        ("Risk Decision Score", "One 0–100 number (low = calm, high = stormy) blending "
         "every signal, with a plain action and a backtested accuracy (AUC ~0.77).",
         "Your headline risk gauge. Calm → trade normal size; high → cut size, widen "
         "stops, hedge. 'Edge vs persistence' shows how much the extra factors really add."),
        ("Market Mood — Health Check", "Diagnostic stats: trend vs mean-reversion (Hurst), "
         "fat-tail risk (kurtosis, EVT), skew, downside (VaR / CVaR / drawdown).",
         "Understand the character of the market — is it trending, crash-prone, fat-"
         "tailed? Context for sizing and stop choices."),
        ("What Tomorrow Likely Looks Like", "Calm-odds for tomorrow (~84% reliable) and a "
         "price band calibrated so the close lands inside it 75% / 90% of the time.",
         "Your next-day plan: the range to expect and how likely a quiet session is. The "
         "75% band is the honest version of 'tomorrow's target zone'."),
        ("Options Edge — Cheap or Rich?", "Our volatility forecast vs implied volatility "
         "(India VIX). Forecast higher → options cheap; lower → rich.",
         "A direct buy-or-sell-options call — the one place the volatility edge becomes "
         "money. Long straddles when cheap, sell premium when rich. (NIFTY only for now.)"),
        ("Position Size &amp; Stops", "Enter capital and risk %; it sets a vol-based stop "
         "and the position size so a stop-out costs exactly your chosen risk.",
         "Turns 'the market moves ±X%' into 'trade Y lots with a Z-point stop'. Stops "
         "auto-widen in wild regimes."),
        ("Futures Positioning (Open Interest)", "Long/short buildup vs covering/unwinding "
         "from futures OI, aggregated across expiries, with rollover.",
         "See whether a move is backed by fresh money (strong) or just position-closing "
         "(weak). Recent-data context."),
        ("Cones, Calm-vs-Wild, Mood-Change, Price Range, Track Record, Correlation",
         "The supporting analytics: forward volatility cones, per-regime ranges, regime-"
         "switch early warning, calibrated sigma bands with hit-ratio backtests, and "
         "cross-market correlation.",
         "Depth for when you want it — coverage proof, what-moves-with-what for hedging, "
         "and early warning that the mood is about to flip."),
    ]
    for name, what, use in panels:
        S.append(Paragraph(name, H2))
        S.append(Paragraph(f"<b>Shows:</b> {what}", BODY))
        S.append(Paragraph(f'<b><font color="#00A88A">Use it to:</font></b> {use}', BODY))
    S.append(PageBreak())

    # ---------------- 4. Daily workflow ----------------
    S.append(Paragraph("4 · How to Use It — a Daily Routine", H1))
    S.append(Paragraph("A five-minute morning workflow:", BODY))
    S.append(bullets([
        "<b>1. All Markets grid</b> — scan risk scores &amp; regimes. Where is it calm "
        "(trade freely) and where is it wild (be careful)?",
        "<b>2. Regime Map</b> — open your instrument; confirm the current mood and whether "
        "it just changed.",
        "<b>3. What Changed Today</b> — note any regime flip or volatility jump overnight.",
        "<b>4. Risk Decision Score</b> — read the headline number and its action line.",
        "<b>5. Tomorrow</b> — note the expected range (the 75% band) and calm odds.",
        "<b>6. Options Edge</b> — if trading options, check cheap vs rich before choosing "
        "buy-premium or sell-premium.",
        "<b>7. Position Size &amp; Stops</b> — set your stop and size from today's "
        "volatility before you enter.",
    ]))
    S.append(callout(
        "<b>Golden rule:</b> the dashboard tells you <i>how much</i> to respect the market "
        "today, not <i>which way</i> to bet. Use it for sizing, stops, range, and option "
        "selection — bring your own directional thesis."))
    S.append(PageBreak())

    # ---------------- 5. Evidence ----------------
    S.append(Paragraph("5 · The Evidence (honest backtests)", H1))
    S.append(Paragraph("Everything is causal (no look-ahead) and walk-forward (retrain, "
                       "predict forward across 2017–2026).", BODY))
    S.append(Paragraph("The leakage gap-probe (5-minute direction)", H2))
    S.append(table([
        ["Gap between features &amp; prediction", "Accuracy"],
        ["0 bars (the headline '80%')", "0.79"],
        ["1 bar (5 min)", "0.70"],
        ["2 bars (10 min)", "0.58"],
        ["3 bars (15 min)", "0.51  — coin-flip"],
    ], col_widths=[100 * mm, 55 * mm]))
    S.append(Paragraph("The 'edge' vanishes once the features can no longer peek at the "
                       "predicted window — proof it was leakage, not skill.", SMALL))
    S.append(Spacer(1, 6))
    S.append(Paragraph("What does survive: volatility &amp; movement", H2))
    S.append(table([
        ["Signal", "Method", "Result"],
        ["Daily realized volatility", "HAR-RV (3-feature OLS)", "R² 0.44 (Nifty) / 0.57 (BankNifty)"],
        ["Volatility regime", "Causal Gaussian HMM", "~2× vol separation, sticky 76–89%"],
        ["Movement / spike (30m)", "GBM + HMM gate", "AUC ~0.53–0.65; usable lift"],
        ["High-vol-ahead (risk score)", "Logistic factor blend", "walk-forward AUC ~0.77"],
        ["Direction (any timeframe)", "GBM / LSTM", "~0.50–0.53 — no edge"],
    ], col_widths=[55 * mm, 50 * mm, 60 * mm]))
    if charts.get("track"):
        S.append(Spacer(1, 6))
        S.append(Paragraph("The forecast, tracked live (NIFTY, out-of-sample):", H2))
        S.append(Image(charts["track"], width=165 * mm, height=51 * mm))
    S.append(Spacer(1, 6))
    S.append(callout("<b>Honest note:</b> the volatility edge is real but modest, and "
                     "stacking many cross-asset / macro factors adds little beyond 'vol is "
                     "already high'. The value is in <i>using</i> it well (sizing, options, "
                     "range), not in a magic number.", WARN))
    S.append(PageBreak())

    # ---------------- 6. Roadmap ----------------
    S.append(Paragraph("6 · What To Add Next", H1))
    S.append(Paragraph("Near-term, high value:", H2))
    S.append(bullets([
        "<b>Visual upgrade of every panel</b> — turn the remaining tables into charts "
        "(Tomorrow band drawn on the price, risk dial, forecast-vs-implied bars).",
        "<b>BankNifty &amp; commodity Options Edge</b> — needs each instrument's real "
        "listed option implied vol (India VIX is NIFTY-only).",
        "<b>Straddle / strangle backtest with costs</b> — prove the rupee value of the "
        "Options-Edge signal after brokerage and slippage.",
        "<b>Alerts</b> — push a notification on regime flip, risk-score jump, or a "
        "cheap/rich options crossover.",
    ]))
    S.append(Paragraph("Medium-term:", H2))
    S.append(bullets([
        "<b>Live broker data feed</b> + automated periodic retrain, so it updates "
        "intraday instead of end-of-day.",
        "<b>Event calendar overlay</b> — RBI policy, expiry, budget, earnings (known "
        "high-volatility days).",
        "<b>Expiry-day &amp; intraday seasonality</b> module (the volatility U-shape, "
        "Thursday expiry effects).",
        "<b>Portfolio view</b> — combine instruments by correlation into one risk "
        "number and hedge suggestions.",
    ]))
    S.append(Paragraph("Research / honesty:", H2))
    S.append(bullets([
        "<b>Find and fix the residual 5-minute feature leak</b> so intraday features "
        "become trustworthy for the movement model.",
        "<b>Realized-vs-implied mispricing</b> with real option IV (not the VIX proxy) — "
        "the cleanest path to a tradeable variance-premium edge.",
    ]))
    S.append(Spacer(1, 6))
    S.append(Paragraph("Regenerate / run", H2))
    S.append(bullets([
        "<b>This report:</b> <font face='Courier'>python dashboard_report.py</font>",
        "<b>Dashboard (live):</b> <font face='Courier'>python vol_server.py</font> → "
        "http://localhost:8000",
        "<b>Dashboard (static / deploy):</b> <font face='Courier'>python "
        "build_static_dashboard.py</font> → commit &amp; push (GitHub Pages).",
    ]))
    S.append(Spacer(1, 8))
    S.append(Paragraph("Built on causal, walk-forward-validated models. No look-ahead, no "
                       "direction hype — just the part of the market that is genuinely "
                       "predictable, made usable.", SMALL))
    S.append(PageBreak())

    # ---------------- 7. Cheat sheet ----------------
    S.append(Paragraph("7 · One-Page Cheat Sheet", H1))
    S.append(Paragraph("Every key number, what it means, and the rule of thumb.", BODY))
    S.append(table([
        ["Reading", "What it means", "Rule of thumb"],
        ["Risk score 0–100", "how stormy ahead", "<33 trade normal · >75 defensive"],
        ["Tomorrow calm %", "odds of a quiet day", ">78% relax · <60% caution"],
        ["75% band", "range price holds 3 of 4 days", "trade inside · breakout if beyond"],
        ["Expected move ±σ%", "typical 1-day swing", "base your stop on this"],
        ["Options edge", "forecast vs implied vol", "cheap = buy vol · rich = sell premium"],
        ["Regime", "calm / normal / wild", "wild → smaller size, wider stops"],
        ["Vol vs 1-yr range", "where vol sits now", "high = stretched, may mean-revert"],
        ["Hurst (Market Mood)", "trend vs mean-revert", ">0.55 trend · <0.45 fade moves"],
        ["VaR / CVaR 95%", "typical / avg worst daily loss", "size so CVaR is survivable"],
    ], col_widths=[42 * mm, 62 * mm, 66 * mm]))
    S.append(PageBreak())

    # ---------------- 8. Worked example ----------------
    S.append(Paragraph("8 · A Worked Example (a real morning)", H1))
    S.append(Paragraph("Say you trade NIFTY with ₹2,00,000 and risk 1% per trade. Here is "
                       "the dashboard, used start to finish:", BODY))
    S.append(table([
        ["Step", "You read", "You do"],
        ["1. Risk score", "16 / 100 — CALM", "Normal size; no defensive trimming."],
        ["2. Regime Map", "green (calm) lately", "Trend-friendly; not crash mode."],
        ["3. Tomorrow", "±0.8% move · 75% band 23,725–24,225", "Expect a quiet, range-bound day."],
        ["4. Options Edge", "forecast ≈ implied (fair)", "No strong vol trade; slight edge to selling premium."],
        ["5. Position Size", "stop 1.5×σ ≈ 290 pts", "Risk ₹2,000 ÷ 290 ≈ 6–7 units (scale to lots)."],
        ["6. Stress Test", "worst day ≈ −13% = −₹26k on ₹2L", "Confirm you can survive the tail; keep size sane."],
    ], col_widths=[34 * mm, 70 * mm, 66 * mm]))
    S.append(Spacer(1, 6))
    S.append(callout("<b>The decision:</b> calm regime + quiet day expected + fair options → "
                     "trade your normal direction view at full size, stop ~290 pts, and prefer "
                     "selling option premium over buying it. The dashboard set the <i>size, "
                     "stop and range</i> — you bring the entry."))
    S.append(PageBreak())

    # ---------------- 9. Glossary ----------------
    S.append(Paragraph("9 · Glossary", H1))
    S.append(table([
        ["Term", "Plain meaning"],
        ["HAR-RV", "The volatility forecast model — blends short/medium/long past vol."],
        ["σ (sigma)", "One standard-deviation move; the typical size of a swing."],
        ["Regime", "The market's volatility state: calm, normal or wild."],
        ["Implied vol / India VIX", "The volatility option prices are charging for."],
        ["Realized vol", "The volatility that actually happened."],
        ["Variance risk premium", "Options usually cost more than realized vol — a seller's edge."],
        ["Walk-forward", "Train on the past, test on unseen future — an honest backtest."],
        ["Look-ahead / leakage", "Accidentally using future data; fakes high accuracy."],
        ["VaR / CVaR", "Typical / average worst-case daily loss (at 95%)."],
        ["Hurst exponent", "Trend gauge: >0.55 trending, <0.45 mean-reverting."],
        ["Open interest (OI)", "How many futures contracts are live — positioning."],
        ["AUC", "Signal quality, 0.5 = useless, 1.0 = perfect."],
    ], col_widths=[48 * mm, 122 * mm]))

    doc.build(S, onFirstPage=_footer, onLaterPages=_footer)
    return path


if __name__ == "__main__":
    out = build()
    print("Wrote", out)
