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
import random
import google.generativeai as genai
from datetime import datetime
from collections import Counter
from oauth2client.service_account import ServiceAccountCredentials

# 設定 Matplotlib 後端
matplotlib.use("Agg")

# 嘗試設定中文字體
try:
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'WenQuanYi Micro Hei', 'Arial']
    matplotlib.rcParams['axes.unicode_minus'] = False
except: pass

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
    
    /* Toast 設定 */
    div[data-testid="stToast"] {
        position: fixed !important; top: 60px !important; right: auto !important;
        bottom: auto !important; left: 50% !important; transform: translateX(-50%) !important;
        z-index: 999999 !important; width: auto !important; white-space: nowrap !important;
    }
    div[data-testid="stToast"] > div {
        background-color: #d32f2f !important; color: #FFFFFF !important;
        border-radius: 8px !important; box-shadow: 0px 4px 10px rgba(0,0,0,0.3) !important;
    }
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

# === 3. 我的最愛管理 ===
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
        st.toast(f"✅ {code} 已加入最愛！", icon="⭐")
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
        st.toast(f"🗑️ {code} 已移除", icon="🗑️")
    except: pass

# === 4. 資料映射 ===
@st.cache_data
def get_stock_info_mapping():
    name_map, cat_map = {}, {}
    for f in ["temp_tickers.csv", "tickers.csv"]:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f, dtype=str)
                if 'code' in df.columns:
                    if 'name' in df.columns:
                        name_map.update(dict(zip(df['code'], df['name'])))
                    if 'category' in df.columns:
                        cat_map.update(dict(zip(df['code'], df['category'])))
            except: pass
    return name_map, cat_map

# === 5. 即時繪圖 ===
def update_live_data(codes):
    if not codes: return
    name_map, cat_map = get_stock_info_mapping()
    live_dir = "runs/favorites_live"
    os.makedirs(live_dir, exist_ok=True)
    for f in glob.glob(os.path.join(live_dir, "*.png")):
        try: os.remove(f)
        except: pass
    status = st.empty()
    bar = st.progress(0)
    for i, code in enumerate(codes):
        stock_name = name_map.get(str(code), "")
        stock_cat = cat_map.get(str(code), "")
        safe_name = stock_name.replace("/", "").strip() or "NA"
        safe_cat = stock_cat.replace("/", "").strip() or "一般"
        status.text(f"更新 {code} {stock_name}...")
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
                fname = f"{code}_{safe_name}_{safe_cat}_Live.png"
                mpf.plot(df, type="candle", volume=True, addplot=ap, title=f"{code} {stock_name} ({stock_cat})",
                         savefig=dict(fname=os.path.join(live_dir, fname), dpi=100, bbox_inches="tight"))
        except: pass
        bar.progress((i + 1) / len(codes))
    status.empty()
    bar.empty()

# === 6. AI 深度分析 (產業趨勢版) ===
def analyze_stock_with_ai(api_key, code):
    if not api_key:
        return "⚠️ 請先在左側欄輸入 Google Gemini API Key。"
    
    try:
        ticker_str = f"{code}.TW"
        stock = yf.Ticker(ticker_str)
        info = stock.info
        
        # 嘗試抓取上櫃
        if 'symbol' not in info:
            ticker_str = f"{code}.TWO"
            stock = yf.Ticker(ticker_str)
            info = stock.info

        # 1. 抓取基礎資訊
        current_price = info.get('currentPrice', '未知')
        sector = info.get('sector', '未知')
        industry = info.get('industry', '未知')
        
        # 2. 抓取新聞
        news_list = stock.news
        news_summary = ""
        if news_list:
            for n in news_list[:5]: # 取前5則
                title = n.get('title', '無標題')
                publisher = n.get('publisher', '未知來源')
                news_summary += f"- {title} ({publisher})\n"
        else:
            news_summary = "近期無重大新聞，請依據產業知識進行分析。"

        # 3. 設定 AI
        genai.configure(api_key=api_key)
        
        prompt = f"""
        角色設定：你是一位資深的產業研究員與投資顧問，擅長挖掘產業趨勢與公司潛在價值。
        任務：請根據以下提供的代號 {code} (所屬產業: {sector}-{industry}) 的相關新聞與數據，結合你的知識庫，進行深度產業分析。

        【參考資訊：近期新聞焦點】
        {news_summary}

        【分析要求】
        請完全依照以下結構輸出，**務必將最重要的結論與原因放在第一段**：

        1. **核心觀點與原因 (Executive Summary)**：
           - **結論**：用一句話總結看法（例如：看好並建議長期持有 / 短線有雜音需觀望 / 產業逆風建議避開）。
           - **關鍵原因**：列出 2-3 點支持上述結論的最主要理由（例如：受惠 AI 伺服器需求爆發、庫存去化結束、新產品將於 Q3 量產等）。

        2. **產業分析 (Industry Analysis)**：
           - **產業地位**：該公司在供應鏈中的角色（上/中/下游）與關鍵競爭優勢。
           - **競爭格局**：目前市場的競爭狀況，以及該公司是否擁有護城河（技術、市佔率、客戶關係）。

        3. **未來展望與機會 (Outlook & Opportunities)**：
           - **成長動能**：未來 1-3 年的主要營收成長來源是什麼？
           - **潛在機會**：是否有新的應用領域、轉型題材或未被市場充分定價的利多？

        請用繁體中文回答，語氣專業且條理分明。
        """

        # 多模型輪詢機制
        candidate_models = [
            "gemini-2.0-flash",       
            "gemini-2.0-flash-exp",   
            "gemini-2.5-flash",       
            "gemini-flash-latest",    
            "gemini-1.5-flash"        
        ]
        
        generated_text = ""
        error_log = []

        for model_name in candidate_models:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                generated_text = response.text
                break 
            except Exception as e:
                error_log.append(f"{model_name}: {str(e)}")
                continue

        if not generated_text:
            return f"❌ 分析失敗。錯誤紀錄: {error_log}"
            
        return generated_text

    except Exception as e:
        return f"❌ 分析失敗: {str(e)}"

# === 7. 畫廊顯示 (包含 AI 按鈕) ===
def get_image_html(file_path, link_url, width="100%"):
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f'<a href="{link_url}" target="_blank"><img src="data:image/png;base64,{data}" style="width:{width}; border-radius:5px;"></a>'

@st.cache_data
def get_stock_category_mapping():
    _, cat_map = get_stock_info_mapping()
    return cat_map

def display_chart_gallery(image_paths, gallery_key):
    if not image_paths:
        st.info("目前無圖表 (請點擊上方按鈕更新行情)。")
        return
    current_faves = get_favorites()
    cat_map = get_stock_category_mapping()
    img_cat_list = [] 
    
    # 建立 (path, cat) 的列表
    for img in image_paths:
        try:
            filename = os.path.basename(img)
            parts = filename.split("_")
            code = parts[0]
            if len(parts) >= 4 and "Live" in filename: cat = parts[2]
            else: cat = cat_map.get(code, "未分類")
        except: cat = "未分類"
        
        if pd.isna(cat) or not cat or str(cat).lower() == 'nan':
            cat = "未分類"
        cat = str(cat)

        img_cat_list.append((img, cat))
        
    cat_counts = Counter(cat for _, cat in img_cat_list)
    total_count = len(image_paths)
    all_option_label = f"全部 ({total_count})"
    display_options = [all_option_label]
    option_map = {all_option_label: "全部"}
    
    for cat in sorted(cat_counts.keys()):
        label = f"{cat} ({cat_counts[cat]})"
        display_options.append(label)
        option_map[label] = cat
        
    c1, c2 = st.columns([2, 2])
    with c1:
        selected_option_label = st.selectbox("🏭 依產品/產業篩選", display_options, key=f"cat_filter_{gallery_key}")
        selected_real_cat = option_map[selected_option_label]
    with c2: items_per_page = st.radio("每頁顯示", [4, 8], horizontal=True, key=f"ipp_{gallery_key}")
    
    # [Fix] 這裡確保 filtered_paths 永遠是 (img, cat) 的 Tuple 列表
    if selected_real_cat == "全部":
        filtered_paths = img_cat_list
    else:
        filtered_paths = [(img, cat) for img, cat in img_cat_list if cat == selected_real_cat]
    
    if not filtered_paths:
        st.warning(f"在分類「{selected_real_cat}」下沒有找到圖片。")
        return
    
    state_key = f"page_idx_{gallery_key}"
    filter_key = f"last_filter_{gallery_key}"
    if filter_key not in st.session_state: st.session_state[filter_key] = all_option_label
    if st.session_state[filter_key] != selected_option_label:
        st.session_state[state_key] = 1 
        st.session_state[filter_key] = selected_option_label
    if state_key not in st.session_state: st.session_state[state_key] = 1
    total_pages = (len(filtered_paths) + items_per_page - 1) // items_per_page
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
    with col_info: st.markdown(f"<div style='text-align: center; line-height: 38px;'><b>{st.session_state[state_key]} / {total_pages}</b></div>", unsafe_allow_html=True)
    
    start_idx = (st.session_state[state_key] - 1) * items_per_page
    current_batch = filtered_paths[start_idx:start_idx + items_per_page]
    cols = st.columns(2 if items_per_page == 4 else 4)
    
    # 取得全域 API Key
    api_key = st.session_state.get('gemini_api_key', '')

    for idx, (img_path, cat) in enumerate(current_batch): # 這裡現在可以正確解包了
        file_name = os.path.basename(img_path)
        try: stock_code = file_name.split("_")[0]
        except: stock_code = "0000"
        
        with cols[idx % (2 if items_per_page == 4 else 4)]:
            st.markdown(get_image_html(img_path, f"https://www.wantgoo.com/stock/{stock_code}/technical-chart"), unsafe_allow_html=True)
            st.caption(f"{file_name}")
            
            # 按鈕列
            b_col1, b_col2 = st.columns([1, 1])
            is_faved = stock_code in current_faves
            
            with b_col1:
                if st.button("★ 已關注" if is_faved else "☆ 加入", key=f"s_{stock_code}_{gallery_key}_{idx}", type="primary" if is_faved else "secondary", use_container_width=True):
                    if is_faved: remove_from_favorites(stock_code)
                    else: add_to_favorites(stock_code)
                    st.rerun()
            
            # AI 分析按鈕
            if "fav_live" in gallery_key or "history" in gallery_key:
                with b_col2:
                    if st.button("🤖 產業診斷", key=f"ai_{stock_code}_{gallery_key}_{idx}", use_container_width=True):
                        if not api_key:
                            st.error("請輸入 API Key")
                        else:
                            with st.spinner(f"正在分析 {stock_code} 的產業趨勢與未來機會..."):
                                analysis = analyze_stock_with_ai(api_key, stock_code)
                                st.session_state[f"ai_res_{stock_code}"] = analysis
            
            # 顯示分析結果 (如果存在)
            if f"ai_res_{stock_code}" in st.session_state:
                with st.expander(f"📊 {stock_code} 產業深度報告", expanded=True):
                    st.markdown(st.session_state[f"ai_res_{stock_code}"])

    st.caption(f"顯示: {selected_option_label} (共 {len(filtered_paths)} 張)")

# === 8. 輔助函式 ===
def get_unique_values(csv_path, col_name):
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, dtype=str)
            if col_name in df.columns: return ["全部"] + sorted(df[col_name].dropna().unique().tolist())
        except: pass
    return ["全部"]

def find_latest_run_dir(root="runs"):
    if not os.path.exists(root): return None
    dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not dirs: return None
    return max(dirs, key=os.path.getmtime)

def get_history_runs(root="runs"):
    if not os.path.exists(root): return []
    dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and "favorites_live" not in d]
    dirs.sort(reverse=True)
    return dirs

def get_subfolders(parent_dir):
    if not os.path.exists(parent_dir): return []
    return [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

# === 9. Sidebar ===
with st.sidebar:
    st.title("🎛️ 掃描控制中心")
    st.caption("TW Scanner Pro (Industry Focus v5.3)")
    
    with st.expander("🔑 AI 設定 (Gemini)", expanded=True):
        gemini_api_key = st.text_input("API Key", type="password", help="請輸入 API Key 以啟用智能分析")
        st.session_state['gemini_api_key'] = gemini_api_key # 存入 Session 供全域使用

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
            "monitor": "純監控", 
            "wave3": "波浪理論 (Wave3)", 
            "ma_entangle": "均線糾結",
            "vol_spike": "爆量 (Spike)", 
            "open_high_low_vol": "開高走低 (OHLV)", 
            "ma_cross": "均線交叉",
            "breakout": "價格突破", 
            "gap": "跳空缺口", 
            "rsi": "RSI",
            "breakout_fade": "過前高+開高走低+爆量 (Breakout Fade)" 
        }
    selected_strats = []
    s_col1, s_col2 = st.columns(2)
    for idx, (key, name) in enumerate(all_strategies.items()):
        col = s_col1 if idx % 2 == 0 else s_col2
        if col.checkbox(name, value=(key=="monitor")): selected_strats.append(key)
    
    enable_intersection = st.checkbox("開啟交集評分", value=True)
    
    # [New] 新增參數設定
    with st.expander("進階參數設定", expanded=False):
        vol_ratio = st.number_input("爆量倍數 (vs 均量)", 1.0, 5.0, 1.5)
        
        # [新增] 今日量 vs 昨日量 (倍數)
        vol_vs_yesterday = st.number_input(
            "爆量: 今日量 vs 昨日量 (倍數)", 
            min_value=0.0, max_value=10.0, value=0.0, step=0.5, 
            help="設定 2.0 代表今日成交量需大於昨日 2 倍。設為 0 代表不啟用此條件。主要用於「爆量 (Spike)」策略。"
        )

        w3_prebreak = st.slider("Wave3 緩衝 %", 0.0, 0.1, 0.03)
        ma_entangle_pct = st.slider("糾結幅度", 0.01, 0.05, 0.02)

    st.header("執行")
    intraday_mode = st.toggle("盤中即時模式", value=False)
    days_lookback = st.number_input("回測天數", value=360)
    run_btn = st.button("🚀 開始掃描", type="primary", use_container_width=True)

# === 10. 掃描執行邏輯 ===
if 'latest_run_dir' not in st.session_state:
    st.session_state.latest_run_dir = find_latest_run_dir()

MOTIVATIONAL_QUOTES = [
    "☕ 巴菲特在喝可樂，你在等訊號，我們都有光明的未來。",
    "🧘 股市虐我千百遍，我待台股如初戀。耐心掃描中...",
    "💎 跌倒了別急著起來，先看地上有沒有便宜的籌碼可以撿。",
    "🚀 機器人正在燃燒 CPU 幫你抓主力的小辮子，請稍候...",
    "🌊 耐心是投資最大的槓桿，等待是為了更精準的狙擊。"
]

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
            cmd.extend(["--wave3-prebreak-pct", str(w3_prebreak), "--wave3-exclude-breakout"])
        if "ma_entangle" in selected_strats:
            cmd.extend(["--ma-entangle-pct", str(ma_entangle_pct)])
        vol_strategies = ["vol_spike", "open_high_low_vol", "breakout_fade"]
        if any(s in selected_strats for s in vol_strategies):
             cmd.extend(["--vol-ratio", str(vol_ratio), "--oh-vol-ratio", str(vol_ratio)])

        # [新增] 傳遞 vol-vs-yesterday 參數
        if vol_vs_yesterday > 0:
            cmd.extend(["--vol-vs-yesterday", str(vol_vs_yesterday)])

        st.info("🚀 掃描啟動中...")
        status_box = st.empty()
        pbar = st.progress(0)
        logs = st.expander("Logs", expanded=True).empty()
        log_lines = []
        captured_dir = None
        
        last_quote_time = time.time()
        st.toast(random.choice(MOTIVATIONAL_QUOTES), icon="💡")

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
                    if "Running" in l: status_box.text(l)
                
                if time.time() - last_quote_time > 5:
                    st.toast(random.choice(MOTIVATIONAL_QUOTES), icon="💬")
                    last_quote_time = time.time()
            
            if process.poll() == 0:
                pbar.progress(100)
                status_box.success(f"✅ 完成！")
                if captured_dir and os.path.exists(captured_dir):
                    st.session_state.latest_run_dir = captured_dir
                else:
                    time.sleep(1)
                    st.session_state.latest_run_dir = find_latest_run_dir()
                st.rerun()
            else: st.error("失敗")
        except Exception as e: st.error(f"Error: {e}")

# === 11. 結果顯示區 ===
if True: 
    st.divider()
    tab1, tab2, tab_fav, tab_ai_tool = st.tabs(["📂 策略明細", "📂 歷史/外部圖庫", "⭐ 我的最愛", "🧠 AI 實驗室"])
    
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
                if os.path.exists(custom_path) and os.path.isdir(custom_path): target_dir = custom_path
                else: st.error("路徑錯誤")
        if target_dir:
            images = glob.glob(os.path.join(target_dir, "*.png"))
            if images:
                st.divider()
                st.markdown(f"**📂 瀏覽:** `{target_dir}` ({len(images)} 張)")
                display_chart_gallery(images, gallery_key=f"history_{os.path.basename(target_dir)}")
            else: st.warning("無 PNG 圖片")

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
            with c_time: st.caption(f"點擊圖片下方的「🤖 產業診斷」查看結果")
            
            live_dir = "runs/favorites_live"
            display_paths = []
            all_live_files = glob.glob(os.path.join(live_dir, "*.png"))
            for code in faves:
                matches = [f for f in all_live_files if os.path.basename(f).startswith(f"{code}_")]
                if matches: display_paths.extend(matches)
            
            if not display_paths: 
                st.warning("⚠️ 請點擊「🔄 手動更新行情」來產生圖片")
            
            display_chart_gallery(display_paths, "fav_live")
        else: st.info("尚無關注股票。")

    with tab_ai_tool:
        st.subheader("🧠 AI 實驗室 (單股查詢)")
        target_code = st.text_input("輸入代號", "2330")
        if st.button("分析"):
            api_key = st.session_state.get('gemini_api_key', '')
            st.markdown(analyze_stock_with_ai(api_key, target_code))