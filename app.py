import sys
import streamlit as st
import pandas as pd
import os
import subprocess
import glob
import time
import base64
from datetime import datetime

# === 1. 頁面設定 (Page Configuration) ===
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
    .metric-card {
        background-color: #262730; border: 1px solid #41444e;
        padding: 15px; border-radius: 8px; color: white;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #1c1e26;
        border-radius: 4px 4px 0px 0px; color: white;
    }
    .stTabs [aria-selected="true"] { background-color: #4CAF50; color: white; }
    /* 讓圖片 hover 時有效果 */
    a img:hover { opacity: 0.8; transition: 0.3s; border: 2px solid #4CAF50; }
</style>
""", unsafe_allow_html=True)

# === 2. 輔助函式 ===
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

def get_image_html(file_path, link_url, width="100%"):
    """將本地圖片轉為 Base64 並包裝成可點擊的 HTML 連結"""
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f'<a href="{link_url}" target="_blank" title="點擊前往玩股網查看詳情"><img src="data:image/png;base64,{data}" style="width:{width}; border-radius:5px;"></a>'

def display_chart_gallery(image_paths, gallery_key):
    """
    通用圖表畫廊函式：改用 Button 翻頁並透過 session_state 記憶頁碼
    """
    if not image_paths:
        st.info("沒有圖表可顯示。")
        return

    # 1. 初始化 Session State (記憶頁碼)
    # 我們用 gallery_key 來區分不同分頁 (例如 top_picks vs strat_xxx) 的頁碼
    state_key = f"page_idx_{gallery_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 1

    # 2. 控制列：選擇每頁張數
    c1, c2 = st.columns([2, 6])
    with c1:
        items_per_page = st.radio("每頁顯示", [4, 8], horizontal=True, key=f"ipp_{gallery_key}")

    # 計算總頁數
    total_images = len(image_paths)
    total_pages = (total_images + items_per_page - 1) // items_per_page
    
    # 防呆：如果切換每頁張數導致當前頁碼超過總頁數，重置為第1頁
    if st.session_state[state_key] > total_pages:
        st.session_state[state_key] = 1

    # 3. 翻頁按鈕區 (上一頁 / 頁碼資訊 / 下一頁)
    col_prev, col_info, col_next = st.columns([1, 2, 1])

    with col_prev:
        # 如果在第1頁，禁用上一頁按鈕
        disable_prev = (st.session_state[state_key] <= 1)
        if st.button("⬅️ 上一頁", key=f"prev_{gallery_key}", disabled=disable_prev, use_container_width=True):
            st.session_state[state_key] -= 1
            st.rerun()

    with col_next:
        # 如果在最後一頁，禁用下一頁按鈕
        disable_next = (st.session_state[state_key] >= total_pages)
        if st.button("下一頁 ➡️", key=f"next_{gallery_key}", disabled=disable_next, use_container_width=True):
            st.session_state[state_key] += 1
            st.rerun()

    with col_info:
        # 居中顯示頁碼資訊
        st.markdown(
            f"<div style='text-align: center; line-height: 38px; font-weight: bold;'>"
            f"第 {st.session_state[state_key]} 頁 / 共 {total_pages} 頁"
            f"</div>", 
            unsafe_allow_html=True
        )

    # 4. 圖片切片與顯示
    current_page = st.session_state[state_key]
    start_idx = (current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    current_batch = image_paths[start_idx:end_idx]

    # 設定 Grid：4張圖用2欄(大)，8張圖用4欄(中)
    cols_count = 2 if items_per_page == 4 else 4
    cols = st.columns(cols_count)

    for idx, img_path in enumerate(current_batch):
        file_name = os.path.basename(img_path)
        try:
            stock_code = file_name.split("_")[0]
        except:
            stock_code = "0000"
        
        wantgoo_url = f"https://www.wantgoo.com/stock/{stock_code}/technical-chart"

        with cols[idx % cols_count]:
            st.markdown(get_image_html(img_path, wantgoo_url), unsafe_allow_html=True)
            st.caption(f"📄 {file_name}")

    st.caption(f"顯示第 {start_idx+1} - {min(end_idx, total_images)} 張，共 {total_images} 張")


# === 3. 左側邊欄：控制中心 ===
with st.sidebar:
    st.title("🎛️ 掃描控制中心")
    st.caption("TW Scanner Pro (Ultimate v2.7)")
    
    # --- A. 資料來源 ---
    st.header("1. 資料來源")
    ticker_file = "tickers.csv"
    uploaded_file = st.file_uploader("上傳股票清單 (CSV)", type=["csv"])
    
    if uploaded_file:
        with open("temp_tickers.csv", "wb") as f: f.write(uploaded_file.getbuffer())
        ticker_path = "temp_tickers.csv"
    elif os.path.exists(ticker_file):
        ticker_path = ticker_file
    else:
        st.error(f"找不到 {ticker_file}，請上傳！")
        ticker_path = None

    if ticker_path:
        groups = get_unique_values(ticker_path, "group")
        categories = get_unique_values(ticker_path, "category")
    else:
        groups, categories = ["全部"], ["全部"]

    selected_group = st.selectbox("🏢 集團歸屬", groups)
    selected_category = st.selectbox("🏭 產業分類", categories)
    min_volume = st.number_input("📊 最低成交量", min_value=0, value=1000000, step=100000, help="例如 1,000,000 代表 1000 張")

    # --- B. 策略選擇 (改為 Checkbox) ---
    st.header("2. 策略選擇")
    all_strategies = {
        "monitor": "純監控 (Monitor)",
        "wave3": "波浪理論 (Wave 3)",
        "ma_entangle": "均線糾結",
        "vol_spike": "爆量 (Vol Spike)",
        "open_high_low_vol": "開高走低",
        "ma_cross": "均線交叉",
        "breakout": "價格突破",
        "gap": "跳空缺口",
        "rsi": "RSI 指標"
    }
    
    selected_strats = []
    st.caption("勾選要執行的策略：")
    
    # 使用兩欄排列 Checkbox 比較省空間
    s_col1, s_col2 = st.columns(2)
    for idx, (key, name) in enumerate(all_strategies.items()):
        col = s_col1 if idx % 2 == 0 else s_col2
        # 預設勾選 Monitor
        if col.checkbox(name, value=(key=="monitor")):
            selected_strats.append(key)
    
    enable_intersection = st.checkbox("開啟交集評分 (Intersection)", value=True)

    # --- C. 進階參數 ---
    with st.expander("⚙️ 進階參數設定", expanded=False):
        if "wave3" in selected_strats:
            st.markdown("**Wave 3 設定**")
            w3_prebreak = st.slider("突破前緩衝區 %", 0.0, 0.1, 0.03, 0.01)
            w3_exclude = st.checkbox("排除已突破", value=True)
        else: w3_prebreak, w3_exclude = 0.03, True

        if "ma_entangle" in selected_strats:
            st.markdown("**均線糾結設定**")
            ma_entangle_pct = st.slider("糾結幅度", 0.01, 0.05, 0.02, 0.01)
        else: ma_entangle_pct = 0.02
            
        if "vol_spike" in selected_strats or "open_high_low_vol" in selected_strats:
            st.markdown("**成交量設定**")
            vol_ratio = st.number_input("爆量倍數", 1.0, 5.0, 1.5, 0.1)
        else: vol_ratio = 1.5

    # --- D. 系統設定 ---
    st.header("3. 執行設定")
    intraday_mode = st.toggle("盤中即時模式", value=False)
    days_lookback = st.number_input("回測天數", value=360)
    run_btn = st.button("🚀 開始掃描", type="primary", use_container_width=True)

# === 4. 主畫面邏輯 ===

if 'latest_run_dir' not in st.session_state:
    st.session_state.latest_run_dir = find_latest_run_dir()

if run_btn:
    if not ticker_path: st.error("請先提供股票清單！")
    elif not selected_strats: st.error("請至少選擇一個策略！")
    else:
        # [修改] 加入 "-u" 參數以強制不快取輸出 (即時顯示用)
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

        # === 進度條與終端機 UI 設置 ===
        status_text = st.empty()
        progress_bar = st.progress(0, text="初始化中...")
        
        # 增加即時 Log 顯示區 (Expander)
        log_expander = st.expander("🖥️ 即時終端機 (Live Logs)", expanded=True)
        with log_expander:
            log_container = st.empty()
        
        logs = []
        
        try:
            # [修改] bufsize=1 代表行緩衝，stderr=subprocess.STDOUT 代表把錯誤也顯示在 log
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1,
                encoding="utf-8"
            )
            
            total_tasks = len(selected_strats) + 1 # 策略數 + 初始化/收尾
            tasks_done = 0
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None: 
                    break
                
                if line:
                    clean_line = line.strip()
                    logs.append(clean_line)
                    # 只顯示最後 10 行避免太長
                    log_container.code("\n".join(logs[-10:]), language="bash")
                    
                    # 簡單的進度判斷邏輯
                    if "Running:" in clean_line or "Running strategy" in clean_line:
                        strat_name = clean_line.split(":")[-1].strip()
                        status_text.info(f"▶️ 正在執行策略: {strat_name}...")
                        tasks_done += 1
                        # 計算百分比
                        pct = int((tasks_done / total_tasks) * 100)
                        progress_bar.progress(min(pct, 95), text=f"執行中: {strat_name}")
                    
                    elif "Building intraday" in clean_line:
                        status_text.warning("⏳ 正在抓取盤中即時數據...")
                        
            
            # 等待進程完全結束
            rc = process.poll()
            
            if rc == 0:
                progress_bar.progress(100, text="✅ 掃描完成")
                status_text.success("✅ 任務全部完成！")
                time.sleep(1) # 給一點時間寫入檔案
                st.session_state.latest_run_dir = find_latest_run_dir()
                st.rerun() # 重新整理以顯示結果
            else:
                status_text.error("❌ 掃描異常終止")
                st.error("掃描器回傳錯誤代碼，請檢查 Log。")

        except Exception as e: 
            st.error(f"執行發生系統錯誤: {e}")

# === 結果渲染 ===
if st.session_state.latest_run_dir:
    run_dir = st.session_state.latest_run_dir
    run_time = os.path.basename(run_dir)
    st.divider()
    st.subheader(f"📊 掃描結果報告 (ID: {run_time})")

    inter_csv = os.path.join(run_dir, "intersection_scored.csv")
    
    if os.path.exists(inter_csv):
        df_res = pd.read_csv(inter_csv)
        c1, c2, c3 = st.columns(3)
        c1.metric("總符合檔數", len(df_res))
        c2.metric("滿分飆股 (Score>=2)", len(df_res[df_res['total_score'] >= 2]))
        c3.metric("最高得分", df_res['total_score'].max() if not df_res.empty else 0)
        
        tab1, tab2 = st.tabs([ "🖼️ 精選圖表 (Top Picks)", "📂 策略明細"])

        with tab1:
            top_chart_dir = os.path.join(run_dir, "charts_intersection_top")
            if os.path.exists(top_chart_dir):
                images = glob.glob(os.path.join(top_chart_dir, "*.png"))
                st.info(f"💡 點擊圖片可開啟玩股網技術分析 (共 {len(images)} 張)")
                # 使用新的畫廊函式
                display_chart_gallery(images, gallery_key="top_picks")
            else:
                st.warning("本次掃描沒有產生高分股 (Score >= 2) 的圖表。")

        with tab2:
            strat_files = glob.glob(os.path.join(run_dir, "*.csv"))
            selected_csv = st.selectbox("選擇策略結果", [f for f in strat_files if "intersection" not in f])
            
            if selected_csv:
                strat_name = os.path.basename(selected_csv).replace(".csv", "")
                strat_chart_dir = os.path.join(run_dir, f"charts_{strat_name}")
                
                if os.path.exists(strat_chart_dir):
                    images = glob.glob(os.path.join(strat_chart_dir, "*.png"))
                    st.divider()
                    st.markdown(f"#### {strat_name} 圖表牆")
                    st.info(f"💡 點擊圖片可開啟玩股網技術分析 (共 {len(images)} 張)")
                    # 使用新的畫廊函式
                    display_chart_gallery(images, gallery_key=f"strat_{strat_name}")
    else:
        st.info("尚無掃描結果")
else:
    st.info("👋 請設定左側參數並開始掃描。")

    # === DEBUG 專用區域 (除錯完可刪除) ===
with st.sidebar.expander("🐞 系統診斷 (Debug Tools)"):
    if st.button("顯示檔案結構"):
        st.write("當前工作目錄:", os.getcwd())
        st.write("目錄下檔案:", os.listdir("."))
        
        if os.path.exists("runs"):
            st.write("runs 資料夾內容:", os.listdir("runs"))
            # 檢查最新的 runs 子資料夾
            latest = find_latest_run_dir()
            if latest:
                st.write(f"最新結果 ({latest}) 內容:", os.listdir(latest))
        else:
            st.error("找不到 runs 資料夾！掃描可能根本沒啟動。")