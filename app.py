import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

# --- 頁面設定 ---
st.set_page_config(page_title="全能股市", layout="wide")

# --- 0. 輔助函式 ---
def get_stock_name(code):
    try:
        clean_code = code.replace('.TW', '').replace('.TWO', '')
        if clean_code in twstock.codes:
            return twstock.codes[clean_code].name
    except: pass
    return code

@st.cache_data(ttl=3600) # 快取機制，避免重複下載
def get_data(stock_code, start_date):
    if not stock_code.endswith('.TW') and not stock_code.endswith('.TWO'):
        stock_code += '.TW'
    try:
        df = yf.download(stock_code, start=start_date, progress=False)
        if df.empty: return None
        
        # 清洗數據
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [col.upper().replace('ADJ CLOSE', 'ADJCLOSE') for col in df.columns]
        target_col = 'CLOSE' if 'CLOSE' in df.columns else 'ADJCLOSE'
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        return df.dropna(subset=[target_col])
    except: return None

# --- 1. 策略邏輯 (保留原核心算法) ---
def run_strategy(df, strategy, capital, stop_loss_pct, enable_range_stop):
    target = 'CLOSE' if 'CLOSE' in df.columns else 'ADJCLOSE'
    
    # 計算指標
    df['MA10'] = df[target].rolling(10).mean()
    df['MA20'] = df[target].rolling(20).mean()
    df['MA60'] = df[target].rolling(60).mean()
    
    # KD
    low_min = df['LOW'].rolling(9).min()
    high_max = df['HIGH'].rolling(9).max()
    rsv = 100 * ((df[target] - low_min) / (high_max - low_min)).fillna(50)
    k_list = []; k=50
    for r in rsv:
        k = (2/3)*k + (1/3)*r; k_list.append(k)
    df['K'] = k_list
    df['Box_Low'] = df['LOW'].rolling(60).min()

    # MACD
    exp12 = df[target].ewm(span=12).mean()
    exp26 = df[target].ewm(span=26).mean()
    df['DIF'] = exp12 - exp26
    df['DEM'] = df['DIF'].ewm(span=9).mean()
    df['MACD_Hist'] = df['DIF'] - df['DEM']

    # 回測迴圈
    position = 0; equity = capital; buy_price = 0
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    history = []
    
    prices = df[target].values; dates = df.index
    
    # 根據策略設定起始點
    start_idx = 60 
    
    for i in range(start_idx, len(df)):
        p = prices[i]; d = dates[i]
        signal_buy = False; signal_sell = False; reason = ""
        
        # --- 策略判斷 ---
        if strategy == "🟢 趨勢 (MA10/60)":
            m10 = df['MA10'].iloc[i]; m60 = df['MA60'].iloc[i]
            if position > 0:
                roi = (p - buy_price)/buy_price
                if roi <= -stop_loss_pct/100: signal_sell=True; reason="停損"
                elif p < m60: signal_sell=True; reason="跌破季線"
            elif position == 0:
                if m10 > m60 and p > m60: signal_buy=True
                
        elif strategy == "🔴 區間 (KD逆勢)":
            k_val = df['K'].iloc[i]; box_low = df['Box_Low'].iloc[i-1]
            if position > 0:
                if enable_range_stop and p < box_low: signal_sell=True; reason="破底停損"
                elif k_val > 80: signal_sell=True; reason="KD超買"
            elif position == 0:
                if k_val < 20: signal_buy=True
                
        elif strategy == "🟡 衝浪 (MACD+MA20)":
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
            position = int(equity / (p * 1.001425))
            if position > 0:
                equity -= position * p * 1.001425
                buy_price = p
                history.append(f"{d.date()} 買進 {p:.1f}")
                buy_x.append(d); buy_y.append(p)

    final_asset = equity
    if position > 0: final_asset += position * prices[-1] * 0.995575
    
    return df, final_asset, history, (buy_x, buy_y, sell_x, sell_y)

# --- 2. 側邊欄 (輸入區) ---
# --- 2. 側邊欄 (輸入區) ---
st.sidebar.title("🎛️ 控制台")

# 定義一個 callback 函數，當輸入框改變時執行
def update_name():
    st.session_state.stock_name = get_stock_name(st.session_state.stock_input)

# 輸入框綁定 key 和 on_change
stock_input = st.sidebar.text_input(
    "股票代碼", 
    value="2382", 
    max_chars=10, 
    key="stock_input", 
    on_change=update_name
)

# 初始化 session_state (第一次執行時)
if 'stock_name' not in st.session_state:
    st.session_state.stock_name = get_stock_name("2382")

# 顯示目前標的 (直接讀取最新的 state)
st.sidebar.info(f"目前標的：{stock_input} {st.session_state.stock_name}")

strategy = st.sidebar.radio("選擇戰略", ["🟢 趨勢 (MA10/60)", "🔴 區間 (KD逆勢)", "🟡 衝浪 (MACD+MA20)"])

# 進階設定
with st.sidebar.expander("⚙️ 參數與資金設定", expanded=True):
    capital = st.number_input("初始本金", value=450000, step=10000)
    start_date = st.date_input("回測開始日", value=date(2020, 1, 1))
    
    stop_loss = 8.0
    enable_range_stop = False
    
    if strategy == "🟢 趨勢 (MA10/60)":
        stop_loss = st.slider("趨勢停損 %", 2.0, 20.0, 8.0)
    elif strategy == "🔴 區間 (KD逆勢)":
        enable_range_stop = st.checkbox("啟用破底停損 (適合非定存股)", value=False)

# 持倉狀態
st.sidebar.divider()
has_position = st.sidebar.checkbox("我目前持有庫存")
my_cost = 0.0
if has_position:
    my_cost = st.sidebar.number_input("持有成本", value=0.0)

# --- 3. 主畫面 (執行與顯示) ---
st.title(f"📊 全能股市 - {stock_name}")

if st.sidebar.button("🚀 執行分析", type="primary"):
    with st.spinner('正在連線交易所抓取數據...'):
        df = get_data(stock_input, start_date)
    
    if df is not None:
        # 執行策略
        df, final_asset, history, signals = run_strategy(df, strategy, capital, stop_loss, enable_range_stop)
        buy_x, buy_y, sell_x, sell_y = signals
        
        # 計算績效
        total_ret = (final_asset - capital) / capital * 100
        net_profit = final_asset - capital
        
        # --- A. 績效看板 ---
        col1, col2, col3 = st.columns(3)
        col1.metric("最終資產", f"${final_asset:,.0f}")
        col2.metric("總損益", f"${net_profit:,.0f}", f"{total_ret:.2f}%")
        col3.metric("總交易次數", f"{len(history)//2} 次")
        
        # --- B. 互動圖表 (Plotly) ---
        fig = go.Figure()
        
        # K線/股價
        fig.add_trace(go.Scatter(x=df.index, y=df['CLOSE' if 'CLOSE' in df.columns else 'ADJCLOSE'], 
                                 mode='lines', name='股價', line=dict(color='gray', width=1)))
        
        # 策略線圖
        if "趨勢" in strategy:
            fig.add_trace(go.Scatter(x=df.index, y=df['MA10'], name='MA10 (攻)', line=dict(color='orange', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], name='MA60 (守)', line=dict(color='green', width=2)))
        elif "衝浪" in strategy:
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20 (月線)', line=dict(color='blue', width=1.5)))
        elif "區間" in strategy and enable_range_stop:
            fig.add_trace(go.Scatter(x=df.index, y=df['Box_Low'], name='箱底支撐', line=dict(color='red', dash='dash')))

        # 買賣點
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', name='買進', marker=dict(symbol='triangle-up', size=12, color='red')))
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', name='賣出', marker=dict(symbol='triangle-down', size=12, color='green')))

        fig.update_layout(title=f"{stock_input} {stock_name} - {strategy}", height=600, xaxis_rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # --- C. 明日戰術指引 ---
        st.subheader("📋 預測報告")
        last = df.iloc[-1]
        curr_price = last['CLOSE' if 'CLOSE' in df.columns else 'ADJCLOSE']
        
        advice = "無動作"
        color = "grey"
        
        # (這裡簡化重現原本的 advice 邏輯，為了節省篇幅)
        # 你可以把 V24.2 的 advice 判斷邏輯直接貼過來
        if has_position:
            stop_price = my_cost * (1 - stop_loss/100) if my_cost > 0 else 0
            st.info(f"持倉監控中 | 現價: {curr_price:.1f} | 成本: {my_cost}")
            if "趨勢" in strategy and curr_price < last['MA60']:
                advice = "📉 賣出 (跌破季線)"; color="red"
            elif "衝浪" in strategy and curr_price < last['MA20']:
                advice = "📉 賣出 (跌破月線)"; color="red"
            else:
                advice = "✅ 續抱"; color="green"
        else:
            st.info(f"空手觀望中 | 現價: {curr_price:.1f}")
            if "趨勢" in strategy and last['MA10'] > last['MA60'] and curr_price > last['MA60']:
                advice = "⚡ 買進 (黃金交叉)"; color="red"
            else:
                advice = "💤 觀望"; color="gray"

        st.markdown(f"### 指令：:{color}[{advice}]")
        
        # --- D. 交易明細 ---
        with st.expander("查看詳細交易紀錄"):
            for h in history:
                st.text(h)
    else:
        st.error("找不到該股票數據，請確認代碼是否正確。")
