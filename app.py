import sys
import streamlit as st
import pandas as pd
import os
import subprocess
import glob
import time
import base64
import gspread
import yfinance as yf
import mplfinance as mpf
import matplotlib
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials

# 設定 Matplotlib 後端
matplotlib.use("Agg")

# === 1. 頁面設定 ===
st.set_page_config(
    page_title="TW Scanner Pro 戰情室",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === 自定義 CSS ===
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #1c1e26;
        border-radius: 4px 4px 0px 0px; color: white;
    }
    .stTabs [aria-selected="true"] { background-color: #4CAF50; color: white; }
    a img:hover { opacity: 0.8; transition: 0.3s; border: 2px solid #4CAF50; }
</style>
""", unsafe_allow_html=True)

# === 2. Google Sheets 連線 ===
SHEET_NAME = "Stock_Notes"     
JSON_KEY_FILE = "google_key.json" 

@st.cache_resource
def get_google_sheet_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    if os.path.exists(JSON_KEY_FILE):
        try:
            creds = ServiceAccountCredentials.from_json_keyfile_name(JSON_KEY_FILE, scope)
            return gspread.authorize(creds)
        except Exception as e:
            print(f"本機 Key 讀取失敗: {e}")
            return None
    try:
        if "gcp_service_account" in st.secrets:
            creds_dict = st.secrets["gcp_service_account"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            return gspread.authorize(creds)
    except: pass
    return None

def get_sheet(worksheet_name="sheet1"):
    client = get_google_sheet_client()
    if not client: return None
    try:
        sh = client.open(SHEET_NAME)
        if worksheet_name == "Favorites":
            try: return sh.worksheet("Favorites")
            except:
                ws = sh.add_worksheet(title="Favorites", rows="100", cols="5")
                ws.update_cell(1, 1, "code")
                ws.update_cell(1, 2, "added_at")
                return ws
        return sh.sheet1 
    except: return None

# === 3. 我的最愛管理 (快取版) ===
def init_faves_cache():
    if 'faves_cache' not in st.session_state:
        with st.spinner("正在同步關注清單..."):
            st.session_state.faves_cache = fetch_favorites_from_google()

def fetch_favorites_from_google():
    sheet = get_sheet("Favorites")
    if not sheet: return []
    try:
        records = sheet.get_all_records()
        return [str(r['code']) for r in records if str(r['code']).strip()]
    except: return []

def get_favorites():
    init_faves_cache()
    return st.session_state.faves_cache

def add_to_favorites(code):
    sheet = get_sheet("Favorites")
    if not sheet: 
        st.error("無法連線 Google Sheet")
        return
    try:
        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([str(code), updated_at])
        
        if 'faves_cache' in st.session_state:
            if code not in st.session_state.faves_cache:
                st.session_state.faves_cache.append(str(code))
        
        st.toast(f"✅ {code} 已加入最愛！")
    except Exception as e: st.error(f"Error: {e}")

def remove_from_favorites(code):
    sheet = get_sheet("Favorites")
    if not sheet: return
    try:
        cell = sheet.find(str(code))
        if cell: sheet.delete_rows(cell.row)
        
        if 'faves_cache' in st.session_state:
            if str(code) in st.session_state.faves_cache:
                st.session_state.faves_cache.remove(str(code))
        st.toast(f"🗑️ {code} 已移除")
    except: pass

# === 4. 即時繪圖 (產生圖片) ===
def update_live_data(codes):
    if not codes: return
    live_dir = "runs/favorites_live"
    os.makedirs(live_dir, exist_ok=True)
    
    for f in glob.glob(os.path.join(live_dir, "*.png")):
        try: os.remove(f)
        except: pass
        
    status = st.empty()
    bar = st.progress(0)
    
    for i, code in enumerate(codes):
        status.text(f"正在更新 {code} 最新走勢...")
        try:
            ticker = f"{code}.TW"
            df = yf.Ticker(ticker).history(period="1y")
            if df.empty:
                ticker = f"{code}.TWO"
                df = yf.Ticker(ticker).history(period="1y")
            
            if not df.empty:
                df = df.iloc[-120:]
                for w, c in zip([5, 20, 60], ['fuchsia', 'orange', 'green']):
                    df[f'MA{w}'] = df['Close'].rolling(w).mean()
                
                ap = [mpf.make_addplot(df[f"MA{w}"], color=c, width=1) for w,c in zip([5,20,60],['fuchsia','orange','green'])]
                fname = f"{code}_Live.png"
                save_path = os.path.join(live_dir, fname)
                mpf.plot(df, type="candle", volume=True, addplot=ap, title=f"{code} Live", style="yahoo",
                         savefig=dict(fname=save_path, dpi=100, bbox_inches="tight"))
        except: pass
        bar.progress((i + 1) / len(codes))
    
    status.empty()
    bar.empty()

# === 5. 畫廊顯示 (純顯示) ===
def get_image_html(file_path, link_url, width="100%"):
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f'<a href="{link_url}" target="_blank"><img src="data:image/png;base64,{data}" style="width:{width}; border-radius:5px;"></a>'

def display_chart_gallery(image_paths, gallery_key):
    if not image_paths:
        st.info("目前無圖表 (請點擊上方按鈕更新行情)。")
        return

    current_faves = get_favorites() 

    state_key = f"page_idx_{gallery_key}"
    if state_key not in st.session_state: st.session_state[state_key] = 1

    c1, c2 = st.columns([2, 6])
    with c1:
        items_per_page = st.radio("每頁顯示", [4, 8], horizontal=True, key=f"ipp_{gallery_key}")

    total_images = len(image_paths)
    total_pages = (total_images + items_per_page - 1) // items_per_page
    if st.session_state[state_key] > total_pages: st.session_state[state_key] = 1

    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("⬅️ 上一頁", key=f"p_{gallery_key}", disabled=(st.session_state[state_key]<=1), use_container_width=True):
            st.session_state[state_key] -= 1
            st.rerun()
    with col_next:
        if st.button("下一頁 ➡️", key=f"n_{gallery_key}", disabled=(st.session_state[state_key]>=total_pages), use_container_width=True):
            st.session_state[state_key] += 1
            st.rerun()
    with col_info:
        st.markdown(f"<div style='text-align: center; line-height: 38px;'><b>{st.session_state[state_key]} / {total_pages}</b></div>", unsafe_allow_html=True)

    start_idx = (st.session_state[state_key] - 1) * items_per_page
    current_batch = image_paths[start_idx:start_idx + items_per_page]
    cols = st.columns(2 if items_per_page == 4 else 4)

    for idx, img_path in enumerate(current_batch):
        file_name = os.path.basename(img_path)
        try: stock_code = file_name.split("_")[0]
        except: stock_code = "0000"
        wantgoo_url = f"https://www.wantgoo.com/stock/{stock_code}/technical-chart"

        with cols[idx % (2 if items_per_page == 4 else 4)]:
            st.markdown(get_image_html(img_path, wantgoo_url), unsafe_allow_html=True)
            st.caption(f"{file_name}")
            
            is_faved = stock_code in current_faves
            label = "★ 已關注" if is_faved else "☆ 加入關注"
            btype = "primary" if is_faved else "secondary"
            if st.button(label, key=f"s_{stock_code}_{gallery_key}_{idx}", type=btype, use_container_width=True):
                if is_faved: remove_from_favorites(stock_code)
                else: add_to_favorites(stock_code)
                st.rerun()
    st.caption(f"共 {total_images} 張")

# === 6. 輔助函式 ===
def get_unique_values(csv_path, col_name):
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, dtype=str)
            if col_name in df.columns:
                return ["全部"] + sorted(df[col_name].dropna().unique().tolist())
        except: pass
    return ["全部"]

def find_latest_run_dir(root="runs"):
    if not os.path.exists(root): return None
    dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not dirs: return None
    return max(dirs, key=os.path.getmtime)

def get_history_runs(root="runs"):
    if not os.path.exists(root): return []
    dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    dirs = [d for d in dirs if "favorites_live" not in d]
    dirs.sort(reverse=True)
    return dirs

def get_subfolders(parent_dir):
    if not os.path.exists(parent_dir): return []
    return [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

# === 7. Sidebar ===
with st.sidebar:
    st.title("🎛️ 掃描控制中心")
    st.caption("TW Scanner Pro (Ultimate v3.6 Clean)")
    
    ticker_file = "tickers.csv"
    uploaded_file = st.file_uploader("上傳股票清單 (CSV)", type=["csv"])
    if uploaded_file:
        with open("temp_tickers.csv", "wb") as f: f.write(uploaded_file.getbuffer())
        ticker_path = "temp_tickers.csv"
    elif os.path.exists(ticker_file): ticker_path = ticker_file
    else: ticker_path = None

    if ticker_path:
        groups = get_unique_values(ticker_path, "group")
        categories = get_unique_values(ticker_path, "category")
    else: groups, categories = ["全部"], ["全部"]

    selected_group = st.selectbox("🏢 集團歸屬", groups)
    selected_category = st.selectbox("🏭 產業分類", categories)
    min_volume = st.number_input("📊 最低成交量", min_value=0, value=1000000, step=100000)

    st.header("策略選擇")
    all_strategies = {
        "monitor": "純監控", "wave3": "波浪理論", "ma_entangle": "均線糾結",
        "vol_spike": "爆量", "open_high_low_vol": "開高走低", "ma_cross": "均線交叉",
        "breakout": "價格突破", "gap": "跳空缺口", "rsi": "RSI"
    }
    selected_strats = []
    s_col1, s_col2 = st.columns(2)
    for idx, (key, name) in enumerate(all_strategies.items()):
        col = s_col1 if idx % 2 == 0 else s_col2
        if col.checkbox(name, value=(key=="monitor")): selected_strats.append(key)
    
    enable_intersection = st.checkbox("開啟交集評分", value=True)
    with st.expander("進階參數"):
        w3_prebreak = st.slider("Wave3 緩衝 %", 0.0, 0.1, 0.03)
        w3_exclude = st.checkbox("排除已突破", value=True)
        ma_entangle_pct = st.slider("糾結幅度", 0.01, 0.05, 0.02)
        vol_ratio = st.number_input("爆量倍數", 1.0, 5.0, 1.5)

    st.header("執行")
    intraday_mode = st.toggle("盤中即時模式", value=False)
    days_lookback = st.number_input("回測天數", value=360)
    run_btn = st.button("🚀 開始掃描", type="primary", use_container_width=True)

# === 8. 掃描執行邏輯 ===
if 'latest_run_dir' not in st.session_state:
    st.session_state.latest_run_dir = find_latest_run_dir()

if run_btn:
    if not ticker_path: st.error("請提供股票清單")
    elif not selected_strats: st.error("請選擇策略")
    else:
        cmd = [
            sys.executable, "-u", "tw_scanner_pro_final.py",
            "--tickers-file", ticker_path,
            "--strategies", *selected_strats,
            "--min-volume", str(min_volume),
            "--days", str(days_lookback)
        ]
        if enable_intersection: cmd.append("--make-intersection")
        if selected_group != "全部": cmd.extend(["--filter-group", selected_group])
        if selected_category != "全部": cmd.extend(["--filter-category", selected_category])
        if intraday_mode: cmd.append("--intraday-once")
        if "wave3" in selected_strats:
            cmd.extend(["--wave3-prebreak-pct", str(w3_prebreak)])
            if w3_exclude: cmd.append("--wave3-exclude-breakout")
        if "ma_entangle" in selected_strats: cmd.extend(["--ma-entangle-pct", str(ma_entangle_pct)])
        cmd.extend(["--vol-ratio", str(vol_ratio), "--oh-vol-ratio", str(vol_ratio)])

        st.info("🚀 掃描啟動中...")
        status = st.empty()
        pbar = st.progress(0)
        logs = st.expander("Logs", expanded=True).empty()
        log_lines = []
        captured_dir = None
        start_time = time.time()

        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding="utf-8")
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None: break
                if line:
                    l = line.strip()
                    log_lines.append(l)
                    logs.code("\n".join(log_lines[-8:]))
                    if "__OUTPUT_PATH__:" in l:
                        captured_dir = l.split("__OUTPUT_PATH__:")[1].strip()
                        if not os.path.isabs(captured_dir): captured_dir = os.path.join(os.getcwd(), captured_dir)
                    if "Running" in l: status.text(l)
            
            if process.poll() == 0:
                end_time = time.time()
                duration = end_time - start_time
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                time_str = f"{minutes} 分 {seconds} 秒"

                pbar.progress(100)
                status.success(f"✅ 任務全部完成！ (⏱️ 總耗時: {time_str})")
                
                if captured_dir and os.path.exists(captured_dir):
                    st.session_state.latest_run_dir = captured_dir
                else:
                    time.sleep(1)
                    st.session_state.latest_run_dir = find_latest_run_dir()
                st.rerun()
            else: st.error("失敗")
        except Exception as e: st.error(f"Error: {e}")

# === 9. 結果顯示區 (3 Tabs: 策略 -> 歷史 -> 最愛) ===
if True: 
    st.divider()
    # [修改] 移除精選圖表，將最愛移至最後
    tab1, tab2, tab_fav = st.tabs(["📂 策略明細", "📂 歷史/外部圖庫", "⭐ 我的最愛"])

    # --- Tab 1: 策略明細 ---
    with tab1:
        if st.session_state.latest_run_dir:
            run_dir = st.session_state.latest_run_dir
            files = [f for f in glob.glob(os.path.join(run_dir, "*.csv")) if "intersection" not in f]
            sel = st.selectbox("選擇策略", files, format_func=lambda x: os.path.basename(x).replace(".csv",""))
            if sel:
                sname = os.path.basename(sel).replace(".csv", "")
                cdir = os.path.join(run_dir, f"charts_{sname}")
                if os.path.exists(cdir):
                    imgs = glob.glob(os.path.join(cdir, "*.png"))
                    display_chart_gallery(imgs, f"sv_{sname}")
                else: st.warning("無圖表")
        else: st.info("請先執行掃描")

    # --- Tab 2: 歷史/外部圖庫 ---
    with tab2:
        mode = st.radio("模式", ["歷史掃描紀錄", "指定外部路徑"], horizontal=True)
        target_dir = None
        if "歷史" in mode:
            history_runs = get_history_runs()
            if history_runs:
                selected_run_id = st.selectbox("選擇時間", history_runs)
                full_run_path = os.path.join("runs", selected_run_id)
                subfolders = get_subfolders(full_run_path)
                chart_folders = [f for f in subfolders if "charts_" in f]
                if chart_folders:
                    selected_sub = st.selectbox("選擇圖表類型", chart_folders)
                    target_dir = os.path.join(full_run_path, selected_sub)
                else: st.warning("無圖表資料夾")
            else: st.info("無歷史紀錄")
        else:
            custom_path = st.text_input("輸入資料夾絕對路徑")
            if custom_path:
                if os.path.exists(custom_path) and os.path.isdir(custom_path):
                    target_dir = custom_path
                else: st.error("路徑錯誤")

        if target_dir:
            images = glob.glob(os.path.join(target_dir, "*.png"))
            if images:
                st.divider()
                st.markdown(f"**📂 瀏覽:** `{target_dir}` ({len(images)} 張)")
                display_chart_gallery(images, gallery_key=f"history_{os.path.basename(target_dir)}")
            else: st.warning("無 PNG 圖片")

    # --- Tab 3: 我的最愛 (移到最後) ---
    with tab_fav:
        init_faves_cache()
        c_add, c_info = st.columns([1, 3])
        with c_add:
            new_fav = st.text_input("輸入代號 (如 2330)", key="nf")
            if st.button("➕ 加入"):
                if new_fav: 
                    add_to_favorites(new_fav)
                    st.rerun()
        
        faves = get_favorites()
        if faves:
            c_update_btn, c_time = st.columns([1, 3])
            with c_update_btn:
                if st.button("🔄 手動更新行情", type="primary"): 
                    update_live_data(faves)
                    st.rerun()
            with c_time: 
                st.caption(f"顯示資料夾內的圖片 (上次更新時間請見檔名或重新抓取)")
            
            live_dir = "runs/favorites_live"
            display_paths = []
            for code in faves:
                expected_path = os.path.join(live_dir, f"{code}_Live.png")
                if os.path.exists(expected_path):
                    display_paths.append(expected_path)
            
            if not display_paths:
                st.warning("⚠️ 目前還沒有圖片，請點擊上方「🔄 手動更新行情」按鈕來下載最新資料。")
            display_chart_gallery(display_paths, "fav_live")
        else: st.info("尚無關注股票。")