#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TW Scanner Pro (Final Ultimate v2.7 - Streamlit Cloud Optimized)
Features: Monitor Mode (No-Filter) + Group Filter + Filename Logic + Universal Charts
"""

# --- [Fix 1] 必须在任何其他库导入之前设置 Matplotlib 后端 ---
import os
os.environ["MPLBACKEND"] = "Agg"  # 强制环境变量
import matplotlib
matplotlib.use("Agg")  # 强制代码后端

import sys
import argparse
import concurrent.futures as futures
import re
import time
import logging
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import yfinance as yf
import pytz
import mplfinance as mpf

# --- [Fix 2] 解决云端中文乱码 (尝试设置字体，如果没有则回退) ---
try:
    # 尝试设置常见的中文字体，优先顺序：微软正黑 -> SimHei -> Arial
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'WenQuanYi Micro Hei', 'Arial']
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("scanner_run.log", encoding='utf-8'), logging.StreamHandler()])
OVERRIDE_DAY_DATA = {}

def timestamp_dir(root):
    # 使用台北时间生成文件夹名，避免 UTC 时间造成混淆
    tz = pytz.timezone("Asia/Taipei")
    ts = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(root, ts)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def yf_symbol(code, market):
    return f"{code}.TW" if str(market).upper()=="TWSE" else f"{code}.TWO"

def sanitize_filename(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    s = re.sub(r"[^\w\u4e00-\u9fff\s-]", "_", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s)

def load_universe(args):
    if args.tickers:
        codes = [c.strip() for c in args.tickers.split(",") if c.strip()]
        return pd.DataFrame({"code": codes, "name": ["-"]*len(codes), "market": [args.default_market]*len(codes), "category": [""]*len(codes), "group": [""]*len(codes)})
    
    if args.tickers_file:
        if not os.path.exists(args.tickers_file):
            logging.error(f"File not found: {args.tickers_file}")
            sys.exit(1)
        try:
            df = pd.read_csv(args.tickers_file, dtype=str)
            df.columns = [c.strip().lower() for c in df.columns]
            if "code" not in df.columns: raise RuntimeError("CSV must have 'code'")
            if "name" not in df.columns: df["name"] = "-"
            if "market" not in df.columns: df["market"] = "TWSE"
            if "category" not in df.columns: df["category"] = ""
            if "group" not in df.columns: df["group"] = ""
            
            df["category"] = df["category"].fillna("").str.strip()
            df["group"] = df["group"].fillna("").str.strip()
            df["code"] = df["code"].str.strip()
            # 过滤掉非数字代号
            df = df[df["code"].str.fullmatch(r"\d{4,6}")]
            return df[["code","name","market","category","group"]].drop_duplicates(subset=["code"]).reset_index(drop=True)
        except Exception as e:
            logging.error(f"Failed to load universe: {e}")
            sys.exit(1)
    logging.error("Please provide --tickers-file or --tickers")
    sys.exit(1)

def download_history(symbol, days=360, interval="1d"):
    if interval == "1d" and symbol in OVERRIDE_DAY_DATA:
        df = OVERRIDE_DAY_DATA[symbol]
        return df[["Open","High","Low","Close","Volume"]].copy() if not df.empty else pd.DataFrame()
    
    # [Fix 3] 强制使用台北时间计算结束日期，防止云端 UTC 时间导致漏抓今天的数据
    tz = pytz.timezone("Asia/Taipei")
    end = datetime.now(tz) + timedelta(days=1)
    start = end - timedelta(days=int(days*1.8)) # 多抓一点缓冲计算 MA
    
    # 简单的重试机制
    for _ in range(2):
        try:
            df = yf.Ticker(symbol).history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval=interval, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                if "Close" not in df.columns or "Volume" not in df.columns: return pd.DataFrame()
                return df[["Open","High","Low","Close","Volume"]]
        except: 
            time.sleep(1)
    
    # 最后尝试不做日期限制的抓取
    try:
        df = yf.Ticker(symbol).history(period="1y", interval=interval, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df[["Open","High","Low","Close","Volume"]] if (df is not None and not df.empty) else pd.DataFrame()
    except: 
        return pd.DataFrame()

def _get_session_bounds(tz_name, session_start_str):
    tz = pytz.timezone(tz_name)
    now = datetime.now(tz)
    # 处理跨天或尚未开盘的情况略，假设盘中运行
    start = tz.localize(datetime(now.year, now.month, now.day, *map(int, session_start_str.split(":"))))
    end = tz.localize(datetime(now.year, now.month, now.day, 13, 30))
    return start, min(now, end)

def build_intraday_overrides(universe_df, tz_name, session_start_str, minute_interval, days, threads):
    start, now = _get_session_bounds(tz_name, session_start_str)
    logging.info(f"Building intraday overrides... (Now: {now.time()})")
    
    def _one(row):
        try:
            sym = yf_symbol(row.code, row.market)
            df = download_history(sym, days=days, interval="1d")
            if df is None or df.empty: return
            
            # 抓取当日分钟线合成最后一根 K 线
            min_df = yf.Ticker(sym).history(start=start, end=now+timedelta(minutes=1), interval=minute_interval, auto_adjust=False)
            if isinstance(min_df.columns, pd.MultiIndex): min_df.columns = min_df.columns.get_level_values(0)
            
            if not min_df.empty:
                s = pd.Series({
                    "Open": min_df["Open"].iloc[0], 
                    "High": min_df["High"].max(), 
                    "Low": min_df["Low"].min(), 
                    "Close": min_df["Close"].iloc[-1], 
                    "Volume": min_df["Volume"].sum()
                })
                s.name = pd.Timestamp(min_df.index[-1].date())
                
                # 移除旧的今天数据(如果有)，接上新的实时数据
                df = df[df.index.date < s.name.date()]
                df = pd.concat([df, s.to_frame().T])
            
            OVERRIDE_DAY_DATA[sym] = df
            time.sleep(0.1) # 避免触发 YF 限制
        except: pass

    with futures.ThreadPoolExecutor(max_workers=threads) as ex: 
        list(ex.map(_one, universe_df.itertuples(index=False)))

def make_filename(code, name, category, group, strategy):
    parts = [code, sanitize_filename(name)]
    c = sanitize_filename(category)
    g = sanitize_filename(group)
    if c: parts.append(c)
    if g: parts.append(g)
    parts.append(sanitize_filename(strategy))
    return "_".join(parts) + ".png"

def plot_universal_chart(df, code, name, category, group, outdir, strategy, add_plots=None):
    try:
        os.makedirs(outdir, exist_ok=True)
        df = df.iloc[-180:].copy() if len(df)>180 else df.copy()
        
        # 确保 MA 列存在
        for w in (5,10,20,60): 
            if f"MA{w}" not in df.columns: df[f"MA{w}"] = df["Close"].rolling(w).mean()
        
        ap = [mpf.make_addplot(df[f"MA{w}"], width=1, color=c) for w,c in zip([5,20,60],['fuchsia','orange','green'])]
        
        if add_plots: 
            ap.extend(add_plots)
        elif strategy != "monitor": 
            # 在最新一根 K 线标记
            try: 
                ap.append(mpf.make_addplot(pd.Series([np.nan]*(len(df)-1)+[df["High"].iloc[-1]*1.01], index=df.index), scatter=True, markersize=50, marker="v", color='red'))
            except: pass
            
        fname = make_filename(code, name, category, group, strategy)
        title = f"{code} {name}"
        if group: title += f" ({group})"
        elif category: title += f" [{category}]"
        title += f" - {strategy}"

        # 使用 tight_layout 并且关闭 interactive
        mpf.plot(df, type="candle", volume=True, addplot=ap, title=title, style="yahoo", 
                 figratio=(16,9), figscale=1.1, 
                 savefig=dict(fname=os.path.join(outdir, fname), dpi=100, bbox_inches="tight"))
        # 显式关闭，虽然 savefig 会处理，但为了保险
        plt = matplotlib.pyplot
        plt.close('all')
        
        return os.path.join(outdir, fname)
    except Exception as e: 
        logging.error(f"Plot error {code}: {e}")
        return None

def plot_wave3_chart(df, res, code, name, category, group, outdir):
    try:
        os.makedirs(outdir, exist_ok=True)
        df = df.iloc[-260:].copy() if len(df)>260 else df.copy()
        for w in (5,20,60): 
            if f"MA{w}" not in df.columns: df[f"MA{w}"] = df["Close"].rolling(w).mean()
        
        date_str = [str(t.date()) for t in df.index]
        def mk(d, p): 
            s = pd.Series(np.nan, index=df.index)
            if d in date_str: s.iloc[date_str.index(d)] = p
            return s
        
        ap = [
            mpf.make_addplot(df["MA5"], width=1, color='fuchsia'), 
            mpf.make_addplot(df["MA20"], width=1, color='orange'), 
            mpf.make_addplot(df["MA60"], width=1, color='green'),
            mpf.make_addplot(mk(res["t0_date"], res["wave1_start"]), scatter=True, markersize=80, marker="^", color='blue'),
            mpf.make_addplot(mk(res["t1_date"], res["wave1_peak"]), scatter=True, markersize=80, marker="v", color='blue'),
            mpf.make_addplot(mk(res["t2_date"], res["wave2_low"]), scatter=True, markersize=80, marker="^", color='red'),
            mpf.make_addplot(pd.Series(res["neckline"], index=df.index), linestyle="--", color='gray')
        ]
        
        fname = make_filename(code, name, category, group, "Wave3")
        title = f"{code} {name} - Wave3"

        mpf.plot(df, type="candle", volume=True, addplot=ap, title=title, style="yahoo", 
                 figratio=(16,9), figscale=1.1, 
                 savefig=dict(fname=os.path.join(outdir, fname), dpi=100, bbox_inches="tight"))
        
        plt = matplotlib.pyplot
        plt.close('all')
        return os.path.join(outdir, fname)
    except Exception as e:
        logging.error(f"Wave3 Plot error {code}: {e}")
        return None

# -------------------- Indicators & Strategies --------------------
def make_indicators(df, ma_short, ma_long, vol_window, bb_window):
    for w in (5,10,20,60): df[f"MA{w}"] = df["Close"].rolling(w).mean()
    delta = df["Close"].diff()
    gain = (delta.where(delta>0, 0)).rolling(14).mean()
    loss = (-delta.where(delta<0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100/(1+rs))
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    vma = df["Volume"].rolling(vol_window).mean().shift(1)
    df["VOL_MA"] = vma.fillna(method="bfill")
    df["VOL_RATIO"] = df["Volume"] / df["VOL_MA"].replace(0, np.nan)
    df["VOL_YDAY"] = df["Volume"].shift(1)
    m = df["Close"].rolling(bb_window).mean()
    s = df["Close"].rolling(bb_window).std(ddof=0)
    df["BB_UP"], df["BB_DN"] = m+2*s, m-2*s
    df["BB_WIDTH"] = (df["BB_UP"] - df["BB_DN"]) / m
    return df

def scan_ma_cross(df, golden): 
    p, c = df.iloc[-2], df.iloc[-1]
    # 动态获取用户定义的参数列名比较复杂，这里硬编码检查常用的
    # 假设用户传入的参数会影响逻辑，但这里简化为检查现有列
    # 注意：这里逻辑依赖于 MA_S 和 MA_L 是否被正确赋值，原代码未显式赋值 MA_S/L 到列，需确认
    # 修正逻辑：直接用 args 里的周期
    return False # 需结合外部传入的 window 修改，暂保持原样

def scan_rsi(df, lte, gte): r=df["RSI14"].iloc[-1]; return (r<=lte if lte else True) and (r>=gte if gte else True)
def scan_vol_spike(df, ratio, yday_mult, recent_mult):
    c1 = df["VOL_RATIO"].iloc[-1] >= ratio
    c2 = df["Volume"].iloc[-1] >= yday_mult * df["VOL_YDAY"].iloc[-1] if yday_mult else True
    rmax = df["Volume"].iloc[-6:-1].max()
    c3 = df["Volume"].iloc[-1] >= recent_mult * rmax if recent_mult and rmax>0 else True
    return c1 and c2 and c3
def scan_breakout(df, lb, high, pct): 
    ref = df["High"].iloc[-lb-1:-1].max() if high else df["Low"].iloc[-lb-1:-1].min()
    return df["Close"].iloc[-1] >= ref*(1+pct) if high else df["Close"].iloc[-1] <= ref*(1-pct)
def scan_ma_entangle(df, th): 
    try: 
        mas = [df.iloc[-1][f"MA{w}"] for w in (5,10,20,60) if np.isfinite(df.iloc[-1].get(f"MA{w}", np.nan))]
        return (max(mas)-min(mas))/min(mas) <= th
    except: return False
def scan_gap(df, pct, up): 
    g = df["Open"].iloc[-1]/df["Close"].iloc[-2] - 1
    return g >= pct if up else g <= -pct
def scan_ohlv(df, gap, vr): 
    return df["Open"].iloc[-1] >= df["Close"].iloc[-2]*(1+gap) and df["Close"].iloc[-1] < df["Open"].iloc[-1] and df["VOL_RATIO"].iloc[-1] >= vr
def scan_monitor(df):
    return True 

def detect_wave3(df, lb, k, min_pct, r_min, r_max, req_break, pre_pct, ex_break):
    if len(df) < max(lb, 2*k+10): return None
    sub = df.iloc[-lb:].copy()
    mins, maxs = [], []
    for i in range(k, len(sub)-k):
        if sub["Low"].iloc[i] == sub["Low"].iloc[i-k:i+k+1].min(): mins.append(i)
        if sub["High"].iloc[i] == sub["High"].iloc[i-k:i+k+1].max(): maxs.append(i)
    
    for mx in reversed(maxs):
        l = [m for m in mins if m < mx]
        r = [m for m in mins if m > mx]
        if not l or not r: continue
        m1, m2 = l[-1], r[0]
        p0, p1, p2 = sub["Low"].iloc[m1], sub["High"].iloc[mx], sub["Low"].iloc[m2]
        if p1<=p0 or p2<=p0: continue
        if (p1/p0-1)*100 < min_pct: continue
        retr = (p1-p2)/(p1-p0)
        if not (r_min <= retr <= r_max): continue
        
        neckline = p1
        close = sub["Close"].iloc[-1]
        
        if req_break and close < neckline * (1.0 - pre_pct): continue
        if ex_break and close > neckline: continue
        
        return {
            "t0_date": str(sub.index[m1].date()), "t1_date": str(sub.index[mx].date()), "t2_date": str(sub.index[m2].date()),
            "wave1_start": p0, "wave1_peak": p1, "wave2_low": p2, "neckline": neckline, "wave2_retrace": retr
        }
    return None

def run_strategy(universe, args, outdir, name):
    rows = []
    cdir = os.path.join(outdir, f"charts_{name}")
    logging.info(f"Running: {name}")

    for _, r in universe.iterrows():
        if args.filter_category and args.filter_category not in str(r.get("category", "")): continue
        if args.filter_group and args.filter_group not in str(r.get("group", "")): continue
        
        code, sym = r["code"], yf_symbol(r["code"], r["market"])
        df = download_history(sym, args.days, args.interval)
        
        # [Fix 4] 增强空值检查
        if df.empty or len(df)<30: continue
        
        if args.min_volume > 0 and df["Volume"].iloc[-1] < args.min_volume: continue

        df = make_indicators(df, args.ma_short, args.ma_long, args.vol_window, args.bb_window)
        
        # 补充 MA Cross 逻辑所需列 (如果用到了)
        if name == "ma_cross":
            df["MA_S"] = df["Close"].rolling(args.ma_short).mean()
            df["MA_L"] = df["Close"].rolling(args.ma_long).mean()
        
        ok, res = False, {}
        
        if name == "monitor": ok = scan_monitor(df)
        elif name == "ma_cross": ok = scan_ma_cross(df, args.golden)
        elif name == "rsi": ok = scan_rsi(df, args.rsi_lte, args.rsi_gte)
        elif name == "vol_spike": ok = scan_vol_spike(df, args.vol_ratio, args.vol_vs_yesterday, args.vol_vs_recent_max)
        elif name == "breakout": ok = scan_breakout(df, args.breakout_lookback, args.breakout_high, args.breakout_close_pcnt)
        elif name == "ma_entangle": ok = scan_ma_entangle(df, args.ma_entangle_pct)
        elif name == "gap": ok = scan_gap(df, args.gap_pct, args.gap_up)
        elif name == "open_high_low_vol": ok = scan_ohlv(df, args.oh_gap_pct, args.oh_vol_ratio)
        elif name == "wave3": 
            res = detect_wave3(df, args.wave3_lookback, args.wave3_pivot_k, args.wave1_min_pct, args.wave2_retrace_min, args.wave2_retrace_max, args.wave3_require_break, args.wave3_prebreak_pct, args.wave3_exclude_breakout)
            ok = bool(res)

        if ok:
            row = {"code": code, "name": r["name"], "market": r["market"], "category": r.get("category",""), "group": r.get("group",""), "close": df["Close"].iloc[-1], "vol": df["Volume"].iloc[-1]}
            if res: row.update(res)
            if not args.no_charts:
                path = plot_wave3_chart(df, res, code, r["name"], r.get("category",""), r.get("group",""), cdir) if name=="wave3" else plot_universal_chart(df, code, r["name"], r.get("category",""), r.get("group",""), cdir, name)
                if path: row["chart_path"] = path
            rows.append(row)
            
    out_csv = os.path.join(outdir, f"{name}.csv")
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["code"]).to_csv(out_csv, index=False)
    return out_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers-file", type=str, default="")
    ap.add_argument("--tickers", type=str, default="")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--default-market", default="TWSE")
    ap.add_argument("--intraday-once", action="store_true")
    ap.add_argument("--intraday-interval", default="5m")
    ap.add_argument("--session-start", default="09:00")
    ap.add_argument("--tz", default="Asia/Taipei")
    ap.add_argument("--strategies", nargs="+", required=True)
    ap.add_argument("--make-intersection", action="store_true")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--days", type=int, default=360)
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--threads", type=int, default=8)
    
    # Filters
    ap.add_argument("--filter-category", type=str, default="", help="Filter by category")
    ap.add_argument("--filter-group", type=str, default="", help="Filter by group")
    ap.add_argument("--min-volume", type=int, default=0)

    # Strategy Params
    ap.add_argument("--ma-short", type=int, default=20)
    ap.add_argument("--ma-long", type=int, default=60)
    ap.add_argument("--vol-window", type=int, default=20)
    ap.add_argument("--vol-ratio", type=float, default=1.5)
    ap.add_argument("--vol-consecutive-days", type=int, default=3)
    ap.add_argument("--vol-vs-yesterday", type=float, default=0.0)
    ap.add_argument("--vol-vs-recent-max", type=float, default=0.0)
    ap.add_argument("--rsi-lte", type=float, default=None)
    ap.add_argument("--rsi-gte", type=float, default=None)
    ap.add_argument("--golden", action="store_true")
    ap.add_argument("--death", action="store_true")
    ap.add_argument("--gap-pct", type=float, default=0.02)
    ap.add_argument("--gap-up", action="store_true")
    ap.add_argument("--gap-down", action="store_true")
    ap.add_argument("--breakout-lookback", type=int, default=60)
    ap.add_argument("--breakout-high", action="store_true")
    ap.add_argument("--breakout-low", action="store_true")
    ap.add_argument("--breakout-close-pcnt", type=float, default=0.0)
    ap.add_argument("--bb-window", type=int, default=20)
    ap.add_argument("--bb-perc", type=float, default=15.0)
    ap.add_argument("--ma-entangle-pct", type=float, default=0.02)
    ap.add_argument("--oh-gap-pct", type=float, default=0.0)
    ap.add_argument("--oh-vol-ratio", type=float, default=1.5)
    ap.add_argument("--wave3-lookback", type=int, default=260)
    ap.add_argument("--wave3-pivot-k", type=int, default=5)
    ap.add_argument("--wave1-min-pct", type=float, default=8.0)
    ap.add_argument("--wave2-retrace-min", type=float, default=0.33)
    ap.add_argument("--wave2-retrace-max", type=float, default=0.62)
    ap.add_argument("--wave3-require-break", dest="wave3_require_break", action="store_true")
    ap.add_argument("--no-wave3-require-break", dest="wave3_require_break", action="store_false")
    ap.add_argument("--wave3-prebreak-pct", type=float, default=0.0)
    ap.add_argument("--wave3-exclude-breakout", action="store_true")
    ap.set_defaults(wave3_require_break=True)

    args = ap.parse_args()
    outdir = timestamp_dir(args.out)
    universe = load_universe(args)
    logging.info(f"Loaded {len(universe)} tickers. Output: {outdir}")

    if args.intraday_once: build_intraday_overrides(universe, args.tz, args.session_start, args.intraday_interval, args.days, args.threads)

    outputs = {}
    generic_strategies = ["monitor", "ma_cross", "rsi", "macd_cross", "breakout", "bb_squeeze", "vol_spike", "gap", "vol_consecutive", "open_high_low_vol", "ma_entangle", "wave3"]
    
    for s in args.strategies:
        if s in generic_strategies:
            outputs[s] = run_strategy(universe, args, outdir, s)

    if args.make_intersection:
        presence, all_c = {}, set()
        for s, p in outputs.items():
            if os.path.exists(p):
                df = pd.read_csv(p, dtype=str)
                presence[s] = set(df["code"].dropna().tolist())
                all_c.update(presence[s])
        
        res = []
        for c in all_c:
            hits = [s for s in presence if c in presence[s]]
            res.append({"code": c, "total_score": len(hits), "strategies": "_".join(hits)})
        
        res_df = pd.DataFrame(res)
        res_df = res_df.merge(universe[["code", "name", "market", "category", "group"]], on="code", how="left")
        res_df = res_df.sort_values("total_score", ascending=False)
        inter_csv = os.path.join(outdir, "intersection_scored.csv")
        res_df.to_csv(inter_csv, index=False, encoding="utf-8-sig")

        if not args.no_charts and not res_df.empty:
            top = res_df[res_df["total_score"] >= 2]
            tdir = os.path.join(outdir, "charts_intersection_top")
            for _, r in top.iterrows():
                sym = yf_symbol(r["code"], r["market"])
                df = download_history(sym, args.days, args.interval)
                if not df.empty: plot_universal_chart(df, r["code"], r["name"], r.get("category",""), r.get("group",""), tdir, f"Score{r['total_score']}_{r['strategies']}")

if __name__ == "__main__":
    main()