import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

# --- 安全導入 twstock ---
try:
    import twstock
    HAS_TWSTOCK = True
except ImportError:
    HAS_TWSTOCK = False

st.set_page_config(page_title="全能股市指揮官 V33", layout="wide")

# ==========================================
# 0. 核心數據引擎 (共用)
# ==========================================
def get_stock_name(code):
    if HAS_TWSTOCK:
        try:
            clean_code = code.replace('.TW', '').replace('.TWO', '')
            if clean_code in twstock.codes:
                return twstock.codes[clean_code].name
        except: pass
    try:
        ticker = yf.Ticker(code if code.endswith('.TW') else code + '.TW')
        return ticker.info.get('shortName', code)
    except: return code

@st.cache_data(ttl=300) 
def get_data(stock_code, start_date):
    if not stock_code.endswith('.TW') and not stock_code.endswith('.TWO'):
        stock_code += '.TW'
    
    # 自動補正日期：至少抓 180 天以確保 MA60 能計算
    days_diff = (date.today() - start_date).days
    if days_diff < 180: 
        start_date = date.today() - timedelta(days=200)
        
    try:
        df = yf.download(stock_code, start=start_date, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [col.upper().replace('ADJ CLOSE', 'ADJCLOSE') for col in df.columns]
        target_col = 'CLOSE' if 'CLOSE' in df.columns else 'ADJCLOSE'
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        return df.dropna(subset=[target_col])
    except: return None

# ==========================================
# 1. 策略計算引擎 (共用)
# ==========================================
def calculate_indicators(df):
    target = 'CLOSE' if 'CLOSE' in df.columns else 'ADJCLOSE'
    # 均線
    df['MA10'] = df[target].rolling(10).mean()
    df['MA20'] = df[target].rolling(20).mean()
    df['MA60'] = df[target].rolling(60).mean()
    
    # KD
    low_9 = df['LOW'].rolling(9).min(); high_9 = df['HIGH'].rolling(9).max()
    rsv = 100 * ((df[target] - low_9) / (high_9 - low_9)).fillna(50)
    k_list = []; k=50
    for r in rsv:
        k = (2/3)*k + (1/3)*r; k_list.append(k)
    df['K'] = k_list
    df['Box_Low'] = df['LOW'].rolling(60).min()

    # MACD
    exp12 = df[target].ewm(span=12).mean(); exp26 = df[target].ewm(span=26).mean()
    df['DIF'] = exp12 - exp26
    df['DEM'] = df['DIF'].ewm(span=9).mean()
    df['MACD_Hist'] = df['DIF'] - df['DEM']
    return df

def run_backtest(df, strategy, capital, stop_loss_pct, take_profit_pct, enable_range_stop):
    if capital <= 0: capital = 10000 
    target = 'CLOSE' if 'CLOSE' in df.columns else 'ADJCLOSE'
    
    position = 0; equity = capital; buy_price = 0
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    history = []
    prices = df[target].values; dates = df.index
    
    start_idx = 60 
    
    for i in range(start_idx, len(df)):
        p = prices[i]; d = dates[i]
        signal_buy = False; signal_sell = False; reason = ""
        
        # --- 策略邏輯判斷 ---
        if "趨勢" in strategy:
            m10 = df['MA10'].iloc[i]; m60 = df['MA60'].iloc[i]
            if position > 0:
                roi = (p - buy_price)/buy_price
                if roi <= -stop_loss_pct/100: signal_sell=True; reason="停損"
                elif p < m60: signal_sell=True; reason="跌破季線"
            elif position == 0:
                if m10 > m60 and p > m60: signal_buy=True

        elif "快攻" in strategy:
            m60 = df['MA60'].iloc[i]; prev_p = prices[i-1]; prev_m60 = df['MA60'].iloc[i-1]
            if position > 0:
                roi = (p - buy_price)/buy_price
                if roi >= take_profit_pct/100: signal_sell=True; reason=f"停利 (+{take_profit_pct}%)"
                elif roi <= -stop_loss_pct/100: signal_sell=True; reason="停損"
                elif p < m60: signal_sell=True; reason="跌破季線"
            elif position == 0:
                if p > m60 and prev_p < prev_m60: signal_buy=True
                
        elif "區間" in strategy:
            k_val = df['K'].iloc[i]; box_low = df['Box_Low'].iloc[i-1]
            if position > 0:
                if enable_range_stop and p < box_low: signal_sell=True; reason="破底停損"
                elif k_val > 80: signal_sell=True; reason="KD超買"
            elif position == 0:
                if k_val < 20: signal_buy=True
                
        elif "衝浪" in strategy:
            ma20 = df['MA20'].iloc[i]; dif = df['DIF'].iloc[i]; dem = df['DEM'].iloc[i]
            prev_dif = df['DIF'].iloc[i-1]; prev_dem = df['DEM'].iloc[i-1]
            if position > 0:
                dead_cross = (prev_dif > prev_dem) and (dif < dem)
                if dead_cross or p < ma20: signal_sell=True; reason="破線/死叉"
            elif position == 0:
                gold_cross = (prev_dif < prev_dem) and (dif > dem)
                if gold_cross: signal_buy=True

        # --- 執行交易 ---
        if signal_sell and position > 0:
            equity += position * p * 0.995575
            roi = (p - buy_price) / buy_price * 100
            history.append(f"{d.date()} 賣出 {p:.1f} | 獲利 {roi:.1f}% ({reason})")
            sell_x.append(d); sell_y.append(p)
            position = 0
        elif signal_buy and position == 0:
            if p > 0:
                position = int(equity / (p * 1.001425))
                if position > 0:
                    equity -= position * p * 1.001425
                    buy_price = p
                    history.append(f"{d.date()} 買進 {p:.1f}")
                    buy_x.append(d); buy_y.append(p)

    final_asset = equity
    if position > 0: final_asset += position * prices[-1] * 0.995575
    return final_asset, history, (buy_x, buy_y, sell_x, sell_y)

# ==========================================
# 2. 介面佈局
# ==========================================
st.sidebar.title("🎛️ 指揮官控制台 V33")

if 'stock_name' not in st.session_state: st.session_state.stock_name = ""
def update_name(): st.session_state.stock_name = get_stock_name(st.session_state.stock_input)

stock_input = st.sidebar.text_input("股票代碼", value="2382", max_chars=10, key="stock_input", on_change=update_name)
if st.session_state.stock_name == "": st.session_state.stock_name = get_stock_name(stock_input)
st.sidebar.info(f"標的：{stock_input} {st.session_state.stock_name}")

strategy = st.sidebar.radio("選擇戰略", ["🟢 趨勢 (MA10/60)", "🟣 快攻 (突破+停利)", "🔴 區間 (KD逆勢)", "🟡 衝浪 (MACD+MA20)"])

with st.sidebar.expander("⚙️ 策略參數微調", expanded=True):
    stop_loss = 8.0
    take_profit = 20.0
    enable_range_stop = False
    
    if "趨勢" in strategy: stop_loss = st.slider("停損 %", 2.0, 20.0, 8.0)
    elif "快攻" in strategy:
        stop_loss = st.slider("停損 %", 2.0, 20.0, 8.0)
        take_profit = st.slider("🎯 停利目標 %", 5.0, 100.0, 20.0)
    elif "區間" in strategy: enable_range_stop = st.checkbox("啟用破底停損", value=False)

st.sidebar.divider()
st.sidebar.caption("Designed by Gemini for Commander")

# ==========================================
# 3. 主畫面：分頁系統
# ==========================================
st.title(f"📊 全能股市指揮官")

tab1, tab2 = st.tabs(["⚔️ 今日戰情 (操作)", "🧪 歷史回測 (研究)"])

# ------------------------------------------------------------------
# 分頁 1: 今日戰情 (操作)
# ------------------------------------------------------------------
with tab1:
    st.header(f"⚔️ {st.session_state.stock_name} ({stock_input}) - 戰術執行面板")
    
    # 這裡改成 3 欄，加入「買進日期」
    col_pos, col_cost, col_date = st.columns(3)
    has_position = col_pos.checkbox("✅ 我目前持有庫存", value=False)
    
    if has_position:
        my_cost = col_cost.number_input("持有成本 (元)", value=0.0)
        # 預設日期為今天，讓使用者選
        my_buy_date = col_date.date_input("買進日期", value=date.today())
    else:
        my_cost = 0.0
        my_buy_date = None

    if st.button("🚀 掃描今日訊號", type="primary", key="btn_scan"):
        with st.spinner('正在連線交易所獲取最新報價...'):
            df_now = get_data(stock_input, date.today() - timedelta(days=200))
        
        if df_now is not None:
            df_now = calculate_indicators(df_now)
            last = df_now.iloc[-1]
            curr_price = last['CLOSE' if 'CLOSE' in df_now.columns else 'ADJCLOSE']
            
            # 計算持有天數
            days_held_str = ""
            if has_position and my_buy_date:
                days_held = (date.today() - my_buy_date).days
                days_held_str = f"(已持有 {days_held} 天)"
            
            # --- 數據驗證區 ---
            with st.expander("🔍 數據驗證"):
                st.write(f"**資料日期：** {last.name.date()}")
                c1, c2, c3 = st.columns(3)
                c1.metric("最新收盤價", f"{curr_price:.2f}")
                c2.metric("MA60 (季線)", f"{last['MA60']:.2f}")
                if "趨勢" in strategy: c3.metric("MA10", f"{last['MA10']:.2f}")
                if "區間" in strategy: c3.metric("K值", f"{last['K']:.2f}")
                if "衝浪" in strategy: c3.metric("MA20", f"{last['MA20']:.2f}")

            # --- 訊號判讀邏輯 ---
            advice = "無動作"; color = "grey"; details = ""
            
            # 判斷持有資訊字串
            holding_info = ""
            if has_position:
                 holding_info = f" | 買進日: {my_buy_date} {days_held_str}"

            if "趨勢" in strategy:
                if has_position:
                    stop_price = my_cost * (1 - stop_loss/100) if my_cost > 0 else 0
                    if curr_price <= stop_price: advice = "🛑 停損賣出"; color = "red"; details = f"觸發 {stop_loss}% 停損"
                    elif curr_price < last['MA60']: advice = "📉 趨勢轉弱賣出"; color = "red"; details = "收盤跌破季線"
                    else: advice = "✅ 續抱"; color = "green"; details = "趨勢向上且未達停損" + holding_info
                else:
                    if last['MA10'] > last['MA60'] and curr_price > last['MA60']: advice = "⚡ 買進"; color = "red"; details = "MA10 黃金交叉 MA60"
                    else: advice = "💤 觀望"; color = "gray"; details = "等待均線交叉"

            elif "快攻" in strategy:
                if has_position:
                    tp_price = my_cost * (1 + take_profit/100)
                    sl_price = my_cost * (1 - stop_loss/100)
                    if curr_price >= tp_price: advice = "💰 獲利了結"; color = "green"; details = f"達成 {take_profit}% 停利目標"
                    elif curr_price <= sl_price: advice = "🛑 停損賣出"; color = "red"; details = f"觸發 {stop_loss}% 停損"
                    elif curr_price < last['MA60']: advice = "📉 破線賣出"; color = "red"; details = "跌破季線防守點"
                    else: advice = "✅ 續抱"; color = "green"; details = "未達停利/停損點" + holding_info
                else:
                    prev_p = df_now['CLOSE' if 'CLOSE' in df_now.columns else 'ADJCLOSE'].iloc[-2]
                    prev_m60 = df_now['MA60'].iloc[-2]
                    if curr_price > last['MA60'] and prev_p < prev_m60: advice = "⚡ 買進"; color = "red"; details = "股價強勢突破季線"
                    else: advice = "💤 觀望"; color = "gray"; details = "等待突破季線"
            
            # 其他策略...
            elif "區間" in strategy and not has_position and last['K'] < 20: advice = "⚡ 買進"; color="red"; details="KD 低檔超賣"
            elif "區間" in strategy and has_position and last['K'] > 80: advice = "📉 賣出"; color="green"; details="KD 高檔超買" + holding_info
            elif "衝浪" in strategy and not has_position and last['MACD_Hist'] > 0 and df_now['MACD_Hist'].iloc[-2] < 0: advice = "⚡ 買進"; color="red"; details="MACD 翻紅"
            elif "衝浪" in strategy and has_position and curr_price < last['MA20']: advice = "📉 賣出"; color="red"; details="跌破月線"
            elif has_position: advice = "✅ 續抱"; color="green"; details="未出現賣訊" + holding_info
            else: advice = "💤 觀望"; color="gray"; details="無進場訊號"

            # --- 顯示巨大指令卡 ---
            st.divider()
            st.markdown(f"<h1 style='text-align: center; color: {color};'>{advice}</h1>", unsafe_allow_html=True)
            st.info(f"ℹ️ 戰術詳情: {details}") # 改用 info 顯示詳情
            st.divider()
            
            # 畫圖 (標示買進點)
            fig_now = go.Figure()
            fig_now.add_trace(go.Scatter(x=df_now.index, y=df_now['CLOSE' if 'CLOSE' in df_now.columns else 'ADJCLOSE'], mode='lines', name='股價', line=dict(color='gray')))
            fig_now.add_trace(go.Scatter(x=df_now.index, y=df_now['MA60'], mode='lines', name='MA60', line=dict(color='green', width=2)))
            
            # 如果有設定買進日期，在地圖上畫一個點標記
            if has_position and my_buy_date:
                # 找到最接近買進日的數據點
                try:
                    buy_point = df_now.loc[df_now.index >= pd.Timestamp(my_buy_date)].iloc[0]
                    buy_date_real = buy_point.name
                    buy_price_real = my_cost if my_cost > 0 else buy_point['CLOSE' if 'CLOSE' in df_now.columns else 'ADJCLOSE']
                    
                    fig_now.add_trace(go.Scatter(
                        x=[buy_date_real], y=[buy_price_real],
                        mode='markers+text', name='您的買點',
                        marker=dict(symbol='star', size=15, color='gold', line=dict(width=2, color='black')),
                        text=['您的買點'], textposition='top center'
                    ))
                except:
                    pass # 如果日期太早，圖表畫不出來就算了

            if "趨勢" in strategy: fig_now.add_trace(go.Scatter(x=df_now.index, y=df_now['MA10'], mode='lines', name='MA10', line=dict(color='orange')))
            fig_now.update_layout(height=400, title="近期走勢圖", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_now, use_container_width=True)

        else:
            st.error("❌ 讀取失敗。")

# ------------------------------------------------------------------
# 分頁 2: 歷史回測
# ------------------------------------------------------------------
with tab2:
    st.header("🧪 歷史戰略研發室")
    col_cap, col_date = st.columns(2)
    capital = col_cap.number_input("回測本金", value=450000, step=10000)
    start_date = col_date.date_input("回測開始日", value=date(2020, 1, 1))
    
    if st.button("📊 執行完整回測", key="btn_backtest"):
        with st.spinner('正在進行歷史推演...'):
            df_hist = get_data(stock_input, start_date)
        
        if df_hist is not None:
            df_hist = calculate_indicators(df_hist)
            safe_capital = capital if capital > 0 else 10000
            
            final_asset, history, signals = run_backtest(df_hist, strategy, safe_capital, stop_loss, take_profit, enable_range_stop)
            buy_x, buy_y, sell_x, sell_y = signals
            
            total_ret = (final_asset - safe_capital) / safe_capital * 100
            net_profit = final_asset - safe_capital
            
            m1, m2, m3 = st.columns(3)
            m1.metric("最終資產", f"${final_asset:,.0f}")
            m2.metric("總損益", f"${net_profit:,.0f}", f"{total_ret:.2f}%")
            m3.metric("交易次數", f"{len(history)//2} 次")
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(x=df_hist.index, y=df_hist['CLOSE' if 'CLOSE' in df_hist.columns else 'ADJCLOSE'], mode='lines', name='股價', line=dict(color='gray', alpha=0.5)))
            fig_hist.add_trace(go.Scatter(x=df_hist.index, y=df_hist['MA60'], mode='lines', name='季線', line=dict(color='green')))
            fig_hist.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', name='買進', marker=dict(symbol='triangle-up', size=8, color='red')))
            fig_hist.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', name='賣出', marker=dict(symbol='triangle-down', size=8, color='green')))
            fig_hist.update_layout(height=500, title=f"完整歷史回測")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            with st.expander("查看詳細交易紀錄"):
                for h in history: st.text(h)
