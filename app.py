{\rtf1\ansi\ansicpg950\cocoartf2865
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import yfinance as yf\
import pandas as pd\
import numpy as np\
import plotly.graph_objects as go\
from datetime import date, timedelta\
import twstock\
\
# --- \uc0\u38913 \u38754 \u35373 \u23450  ---\
st.set_page_config(page_title="\uc0\u20840 \u33021 \u32929 \u24066 \u25351 \u25582 \u23448 ", layout="wide")\
\
# --- 0. \uc0\u36628 \u21161 \u20989 \u24335  ---\
def get_stock_name(code):\
    try:\
        clean_code = code.replace('.TW', '').replace('.TWO', '')\
        if clean_code in twstock.codes:\
            return twstock.codes[clean_code].name\
    except: pass\
    return code\
\
@st.cache_data(ttl=3600) # \uc0\u24555 \u21462 \u27231 \u21046 \u65292 \u36991 \u20813 \u37325 \u35079 \u19979 \u36617 \
def get_data(stock_code, start_date):\
    if not stock_code.endswith('.TW') and not stock_code.endswith('.TWO'):\
        stock_code += '.TW'\
    try:\
        df = yf.download(stock_code, start=start_date, progress=False)\
        if df.empty: return None\
        \
        # \uc0\u28165 \u27927 \u25976 \u25818 \
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)\
        df.columns = [col.upper().replace('ADJ CLOSE', 'ADJCLOSE') for col in df.columns]\
        target_col = 'CLOSE' if 'CLOSE' in df.columns else 'ADJCLOSE'\
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')\
        return df.dropna(subset=[target_col])\
    except: return None\
\
# --- 1. \uc0\u31574 \u30053 \u37007 \u36655  (\u20445 \u30041 \u21407 \u26680 \u24515 \u31639 \u27861 ) ---\
def run_strategy(df, strategy, capital, stop_loss_pct, enable_range_stop):\
    target = 'CLOSE' if 'CLOSE' in df.columns else 'ADJCLOSE'\
    \
    # \uc0\u35336 \u31639 \u25351 \u27161 \
    df['MA10'] = df[target].rolling(10).mean()\
    df['MA20'] = df[target].rolling(20).mean()\
    df['MA60'] = df[target].rolling(60).mean()\
    \
    # KD\
    low_min = df['LOW'].rolling(9).min()\
    high_max = df['HIGH'].rolling(9).max()\
    rsv = 100 * ((df[target] - low_min) / (high_max - low_min)).fillna(50)\
    k_list = []; k=50\
    for r in rsv:\
        k = (2/3)*k + (1/3)*r; k_list.append(k)\
    df['K'] = k_list\
    df['Box_Low'] = df['LOW'].rolling(60).min()\
\
    # MACD\
    exp12 = df[target].ewm(span=12).mean()\
    exp26 = df[target].ewm(span=26).mean()\
    df['DIF'] = exp12 - exp26\
    df['DEM'] = df['DIF'].ewm(span=9).mean()\
    df['MACD_Hist'] = df['DIF'] - df['DEM']\
\
    # \uc0\u22238 \u28204 \u36852 \u22280 \
    position = 0; equity = capital; buy_price = 0\
    buy_x, buy_y, sell_x, sell_y = [], [], [], []\
    history = []\
    \
    prices = df[target].values; dates = df.index\
    \
    # \uc0\u26681 \u25818 \u31574 \u30053 \u35373 \u23450 \u36215 \u22987 \u40670 \
    start_idx = 60 \
    \
    for i in range(start_idx, len(df)):\
        p = prices[i]; d = dates[i]\
        signal_buy = False; signal_sell = False; reason = ""\
        \
        # --- \uc0\u31574 \u30053 \u21028 \u26039  ---\
        if strategy == "\uc0\u55357 \u57314  \u36264 \u21218  (MA10/60)":\
            m10 = df['MA10'].iloc[i]; m60 = df['MA60'].iloc[i]\
            if position > 0:\
                roi = (p - buy_price)/buy_price\
                if roi <= -stop_loss_pct/100: signal_sell=True; reason="\uc0\u20572 \u25613 "\
                elif p < m60: signal_sell=True; reason="\uc0\u36300 \u30772 \u23395 \u32218 "\
            elif position == 0:\
                if m10 > m60 and p > m60: signal_buy=True\
                \
        elif strategy == "\uc0\u55357 \u56628  \u21312 \u38291  (KD\u36870 \u21218 )":\
            k_val = df['K'].iloc[i]; box_low = df['Box_Low'].iloc[i-1]\
            if position > 0:\
                if enable_range_stop and p < box_low: signal_sell=True; reason="\uc0\u30772 \u24213 \u20572 \u25613 "\
                elif k_val > 80: signal_sell=True; reason="KD\uc0\u36229 \u36023 "\
            elif position == 0:\
                if k_val < 20: signal_buy=True\
                \
        elif strategy == "\uc0\u55357 \u57313  \u34909 \u28010  (MACD+MA20)":\
            ma20 = df['MA20'].iloc[i]; dif = df['DIF'].iloc[i]; dem = df['DEM'].iloc[i]\
            prev_dif = df['DIF'].iloc[i-1]; prev_dem = df['DEM'].iloc[i-1]\
            if position > 0:\
                dead_cross = (prev_dif > prev_dem) and (dif < dem)\
                if dead_cross or p < ma20: signal_sell=True; reason="\uc0\u30772 \u32218 /\u27515 \u21449 "\
            elif position == 0:\
                gold_cross = (prev_dif < prev_dem) and (dif > dem)\
                if gold_cross: signal_buy=True\
\
        # --- \uc0\u22519 \u34892 \u20132 \u26131  ---\
        if signal_sell and position > 0:\
            equity += position * p * 0.995575\
            roi = (p - buy_price) / buy_price * 100\
            history.append(f"\{d.date()\} \uc0\u36067 \u20986  \{p:.1f\} | \u29554 \u21033  \{roi:.1f\}% (\{reason\})")\
            sell_x.append(d); sell_y.append(p)\
            position = 0\
            \
        elif signal_buy and position == 0:\
            position = int(equity / (p * 1.001425))\
            if position > 0:\
                equity -= position * p * 1.001425\
                buy_price = p\
                history.append(f"\{d.date()\} \uc0\u36023 \u36914  \{p:.1f\}")\
                buy_x.append(d); buy_y.append(p)\
\
    final_asset = equity\
    if position > 0: final_asset += position * prices[-1] * 0.995575\
    \
    return df, final_asset, history, (buy_x, buy_y, sell_x, sell_y)\
\
# --- 2. \uc0\u20596 \u37002 \u27396  (\u36664 \u20837 \u21312 ) ---\
st.sidebar.title("\uc0\u55356 \u57243 \u65039  \u25351 \u25582 \u23448 \u25511 \u21046 \u21488 ")\
\
stock_input = st.sidebar.text_input("\uc0\u32929 \u31080 \u20195 \u30908 ", value="2382", max_chars=10)\
stock_name = get_stock_name(stock_input)\
st.sidebar.markdown(f"**\uc0\u30446 \u21069 \u27161 \u30340 \u65306 \{stock_input\} \{stock_name\}**")\
\
strategy = st.sidebar.radio("\uc0\u36984 \u25799 \u25136 \u30053 ", ["\u55357 \u57314  \u36264 \u21218  (MA10/60)", "\u55357 \u56628  \u21312 \u38291  (KD\u36870 \u21218 )", "\u55357 \u57313  \u34909 \u28010  (MACD+MA20)"])\
\
# \uc0\u36914 \u38542 \u35373 \u23450 \
with st.sidebar.expander("\uc0\u9881 \u65039  \u21443 \u25976 \u33287 \u36039 \u37329 \u35373 \u23450 ", expanded=True):\
    capital = st.number_input("\uc0\u21021 \u22987 \u26412 \u37329 ", value=450000, step=10000)\
    start_date = st.date_input("\uc0\u22238 \u28204 \u38283 \u22987 \u26085 ", value=date(2020, 1, 1))\
    \
    stop_loss = 8.0\
    enable_range_stop = False\
    \
    if strategy == "\uc0\u55357 \u57314  \u36264 \u21218  (MA10/60)":\
        stop_loss = st.slider("\uc0\u36264 \u21218 \u20572 \u25613  %", 2.0, 20.0, 8.0)\
    elif strategy == "\uc0\u55357 \u56628  \u21312 \u38291  (KD\u36870 \u21218 )":\
        enable_range_stop = st.checkbox("\uc0\u21855 \u29992 \u30772 \u24213 \u20572 \u25613  (\u36969 \u21512 \u38750 \u23450 \u23384 \u32929 )", value=False)\
\
# \uc0\u25345 \u20489 \u29376 \u24907 \
st.sidebar.divider()\
has_position = st.sidebar.checkbox("\uc0\u25105 \u30446 \u21069 \u25345 \u26377 \u24235 \u23384 ")\
my_cost = 0.0\
if has_position:\
    my_cost = st.sidebar.number_input("\uc0\u25345 \u26377 \u25104 \u26412 ", value=0.0)\
\
# --- 3. \uc0\u20027 \u30059 \u38754  (\u22519 \u34892 \u33287 \u39023 \u31034 ) ---\
st.title(f"\uc0\u55357 \u56522  \u20840 \u33021 \u32929 \u24066 \u25351 \u25582 \u23448  - \{stock_name\}")\
\
if st.sidebar.button("\uc0\u55357 \u56960  \u22519 \u34892 \u25136 \u30053 \u20998 \u26512 ", type="primary"):\
    with st.spinner('\uc0\u27491 \u22312 \u36899 \u32218 \u20132 \u26131 \u25152 \u25235 \u21462 \u25976 \u25818 ...'):\
        df = get_data(stock_input, start_date)\
    \
    if df is not None:\
        # \uc0\u22519 \u34892 \u31574 \u30053 \
        df, final_asset, history, signals = run_strategy(df, strategy, capital, stop_loss, enable_range_stop)\
        buy_x, buy_y, sell_x, sell_y = signals\
        \
        # \uc0\u35336 \u31639 \u32318 \u25928 \
        total_ret = (final_asset - capital) / capital * 100\
        net_profit = final_asset - capital\
        \
        # --- A. \uc0\u32318 \u25928 \u30475 \u26495  ---\
        col1, col2, col3 = st.columns(3)\
        col1.metric("\uc0\u26368 \u32066 \u36039 \u29986 ", f"$\{final_asset:,.0f\}")\
        col2.metric("\uc0\u32317 \u25613 \u30410 ", f"$\{net_profit:,.0f\}", f"\{total_ret:.2f\}%")\
        col3.metric("\uc0\u32317 \u20132 \u26131 \u27425 \u25976 ", f"\{len(history)//2\} \u27425 ")\
        \
        # --- B. \uc0\u20114 \u21205 \u22294 \u34920  (Plotly) ---\
        fig = go.Figure()\
        \
        # K\uc0\u32218 /\u32929 \u20729 \
        fig.add_trace(go.Scatter(x=df.index, y=df['CLOSE' if 'CLOSE' in df.columns else 'ADJCLOSE'], \
                                 mode='lines', name='\uc0\u32929 \u20729 ', line=dict(color='gray', width=1)))\
        \
        # \uc0\u31574 \u30053 \u32218 \u22294 \
        if "\uc0\u36264 \u21218 " in strategy:\
            fig.add_trace(go.Scatter(x=df.index, y=df['MA10'], name='MA10 (\uc0\u25915 )', line=dict(color='orange', width=1)))\
            fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], name='MA60 (\uc0\u23432 )', line=dict(color='green', width=2)))\
        elif "\uc0\u34909 \u28010 " in strategy:\
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20 (\uc0\u26376 \u32218 )', line=dict(color='blue', width=1.5)))\
        elif "\uc0\u21312 \u38291 " in strategy and enable_range_stop:\
            fig.add_trace(go.Scatter(x=df.index, y=df['Box_Low'], name='\uc0\u31665 \u24213 \u25903 \u25744 ', line=dict(color='red', dash='dash')))\
\
        # \uc0\u36023 \u36067 \u40670 \
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', name='\uc0\u36023 \u36914 ', marker=dict(symbol='triangle-up', size=12, color='red')))\
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', name='\uc0\u36067 \u20986 ', marker=dict(symbol='triangle-down', size=12, color='green')))\
\
        fig.update_layout(title=f"\{stock_input\} \{stock_name\} - \{strategy\}", height=600, xaxis_rangeslider_visible=True)\
        st.plotly_chart(fig, use_container_width=True)\
        \
        # --- C. \uc0\u26126 \u26085 \u25136 \u34899 \u25351 \u24341  ---\
        st.subheader("\uc0\u55357 \u56523  \u25351 \u25582 \u23448 \u25136 \u34899 \u22577 \u21578 ")\
        last = df.iloc[-1]\
        curr_price = last['CLOSE' if 'CLOSE' in df.columns else 'ADJCLOSE']\
        \
        advice = "\uc0\u28961 \u21205 \u20316 "\
        color = "grey"\
        \
        # (\uc0\u36889 \u35041 \u31777 \u21270 \u37325 \u29694 \u21407 \u26412 \u30340  advice \u37007 \u36655 \u65292 \u28858 \u20102 \u31680 \u30465 \u31687 \u24133 )\
        # \uc0\u20320 \u21487 \u20197 \u25226  V24.2 \u30340  advice \u21028 \u26039 \u37007 \u36655 \u30452 \u25509 \u36028 \u36942 \u20358 \
        if has_position:\
            stop_price = my_cost * (1 - stop_loss/100) if my_cost > 0 else 0\
            st.info(f"\uc0\u25345 \u20489 \u30435 \u25511 \u20013  | \u29694 \u20729 : \{curr_price:.1f\} | \u25104 \u26412 : \{my_cost\}")\
            if "\uc0\u36264 \u21218 " in strategy and curr_price < last['MA60']:\
                advice = "\uc0\u55357 \u56521  \u36067 \u20986  (\u36300 \u30772 \u23395 \u32218 )"; color="red"\
            elif "\uc0\u34909 \u28010 " in strategy and curr_price < last['MA20']:\
                advice = "\uc0\u55357 \u56521  \u36067 \u20986  (\u36300 \u30772 \u26376 \u32218 )"; color="red"\
            else:\
                advice = "\uc0\u9989  \u32396 \u25265 "; color="green"\
        else:\
            st.info(f"\uc0\u31354 \u25163 \u35264 \u26395 \u20013  | \u29694 \u20729 : \{curr_price:.1f\}")\
            if "\uc0\u36264 \u21218 " in strategy and last['MA10'] > last['MA60'] and curr_price > last['MA60']:\
                advice = "\uc0\u9889  \u36023 \u36914  (\u40643 \u37329 \u20132 \u21449 )"; color="red"\
            else:\
                advice = "\uc0\u55357 \u56484  \u35264 \u26395 "; color="gray"\
\
        st.markdown(f"### \uc0\u25351 \u20196 \u65306 :\{color\}[\{advice\}]")\
        \
        # --- D. \uc0\u20132 \u26131 \u26126 \u32048  ---\
        with st.expander("\uc0\u26597 \u30475 \u35443 \u32048 \u20132 \u26131 \u32000 \u37636 "):\
            for h in history:\
                st.text(h)\
    else:\
        st.error("\uc0\u25214 \u19981 \u21040 \u35442 \u32929 \u31080 \u25976 \u25818 \u65292 \u35531 \u30906 \u35469 \u20195 \u30908 \u26159 \u21542 \u27491 \u30906 \u12290 ")}