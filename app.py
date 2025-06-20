import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from collections import Counter

st.set_page_config(page_title="Portfolio+Options Analyzer", layout="wide")

# ---------- Helper Functions ----------

def format_money(val):
    if val == 0:
        return "۰ دلار"
    elif val >= 1:
        return "{:,.0f} دلار".format(val)
    else:
        return "{:.3f} دلار".format(val).replace('.', '٫')

def format_percent(val):
    return "{:.3f}%".format(val*100).replace('.', '٫')

def format_float(val):
    if abs(val) >= 1:
        return "{:,.3f}".format(val).rstrip('0').rstrip('.')
    else:
        return "{:.6f}".format(val).rstrip('0').rstrip('.')

def read_csv_file(file):
    try:
        file.seek(0)
        df_try = pd.read_csv(file)
        cols_lower = [str(c).strip().lower() for c in df_try.columns]
        if any(x in cols_lower for x in ['date']):
            df = df_try.copy()
        else:
            file.seek(0)
            df = pd.read_csv(file, header=None)
            header_idx = None
            for i in range(min(5, len(df))):
                row = [str(x).strip().lower() for x in df.iloc[i].tolist()]
                if any('date' == x for x in row):
                    header_idx = i
                    break
            if header_idx is None:
                raise Exception("سطر عنوان مناسب (شامل date) یافت نشد.")
            header_row = df.iloc[header_idx].tolist()
            df = df.iloc[header_idx+1:].reset_index(drop=True)
            df.columns = header_row

        date_col = [c for c in df.columns if str(c).strip().lower() == 'date']
        if not date_col:
            raise Exception("ستون تاریخ با نام 'Date' یا مشابه آن یافت نشد.")
        date_col = date_col[0]
        price_candidates = [c for c in df.columns if str(c).strip().lower() in ['price', 'close', 'adj close', 'open']]
        if not price_candidates:
            price_candidates = [c for c in df.columns if c != date_col]
        if not price_candidates:
            raise Exception("ستون قیمت مناسب یافت نشد.")
        price_col = price_candidates[0]
        df = df[[date_col, price_col]].dropna()
        if df.empty:
            raise Exception("پس از حذف داده‌های خالی، داده‌ای باقی نماند.")

        df = df.rename(columns={date_col: "Date", price_col: "Price"})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        if df.empty:
            raise Exception("پس از تبدیل نوع داده، داده معتبری باقی نماند.")
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
        return None

def get_price_dataframe_from_yf(data, t):
    if isinstance(data.columns, pd.MultiIndex):
        if t in data.columns.levels[0]:
            df_t = data[t].reset_index()
            price_col = None
            for col in ['Close', 'Adj Close', 'Open']:
                if col in df_t.columns:
                    price_col = col
                    break
            if price_col is None:
                return None, f"هیچ یک از ستون‌های قیمت (Close, Adj Close, Open) برای {t} پیدا نشد."
            df = df_t[['Date', price_col]].rename(columns={price_col: 'Price'})
            return df, None
        else:
            return None, f"نماد {t} در داده‌های دریافتی وجود ندارد."
    else:
        if 'Date' not in data.columns:
            data = data.reset_index()
        price_col = None
        for col in ['Close', 'Adj Close', 'Open']:
            if col in data.columns:
                price_col = col
                break
        if price_col is None:
            return None, f"هیچ یک از ستون‌های قیمت (Close, Adj Close, Open) برای {t} پیدا نشد."
        df = data[['Date', price_col]].rename(columns={price_col: 'Price'})
        return df, None

def calc_option_return(row_type, price, prev_price, strike, premium, qty):
    if row_type == 'خرید دارایی':
        return (price - prev_price) / prev_price if prev_price != 0 else 0
    elif row_type == 'فروش دارایی':
        return (prev_price - price) / prev_price if prev_price != 0 else 0
    elif row_type == 'خرید کال':
        return (max(price - strike, 0) - premium) / prev_price if prev_price != 0 else 0
    elif row_type == 'فروش کال':
        return (premium - max(price - strike, 0)) / prev_price if prev_price != 0 else 0
    elif row_type == 'خرید پوت':
        return (max(strike - price, 0) - premium) / prev_price if prev_price != 0 else 0
    elif row_type == 'فروش پوت':
        return (premium - max(strike - price, 0)) / prev_price if prev_price != 0 else 0
    else:
        return 0

def calc_options_series(option_rows, prices: pd.Series):
    rets = pd.Series(np.zeros(len(prices)), index=prices.index)
    prev_price = prices.iloc[0]
    for i in range(1, len(prices)):
        price = prices.iloc[i]
        ret_row = 0
        for row in option_rows:
            row_type, strike, premium, qty = row
            ret_row += qty * calc_option_return(row_type, price, prev_price, strike, premium, 1)
        rets.iloc[i] = ret_row
        prev_price = price
    return rets

def efficient_frontier(mean_returns, cov_matrix, points=200, min_weights=None, max_weights=None):
    num_assets = len(mean_returns)
    results = np.zeros((3, points))
    weight_record = []
    for i in range(points):
        for _ in range(100):  # try 100 times to generate valid weights
            w = np.random.dirichlet(np.ones(num_assets), size=1)[0]
            if min_weights is not None:
                if not np.all(w >= min_weights): continue
            if max_weights is not None:
                if not np.all(w <= max_weights): continue
            break
        weights = w
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0,i] = port_std
        results[1,i] = port_return
        results[2,i] = (port_return) / port_std if port_std > 0 else 0
        weight_record.append(weights)
    return results, np.array(weight_record)

def portfolio_risk_return(resampled_returns, weights, freq_label="M"):
    pf_returns = resampled_returns @ weights
    if freq_label == "M":
        ann_factor = 12
    elif freq_label == "W":
        ann_factor = 52
    else:
        ann_factor = 1
    mean_month = pf_returns.mean()
    risk_month = pf_returns.std()
    mean_ann = mean_month * ann_factor
    risk_ann = risk_month * (ann_factor ** 0.5)
    return mean_month, risk_month, mean_ann, risk_ann

# ---------- Session State ----------
if "downloaded_dfs" not in st.session_state:
    st.session_state["downloaded_dfs"] = []
if "uploaded_dfs" not in st.session_state:
    st.session_state["uploaded_dfs"] = []
if "option_rows" not in st.session_state:
    st.session_state["option_rows"] = {}
if "investment_amount" not in st.session_state:
    st.session_state["investment_amount"] = 10000.0

# ---------- Sidebar: Uploads and Download ----------
st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "چند فایل CSV آپلود کنید (هر دارایی یک فایل)", type=['csv'], accept_multiple_files=True, key="uploader"
)
if uploaded_files:
    for file in uploaded_files:
        if not hasattr(file, "uploaded_in_session") or not file.uploaded_in_session:
            df = read_csv_file(file)
            if df is not None:
                st.session_state["uploaded_dfs"].append((file.name.split('.')[0], df))
            file.uploaded_in_session = True

with st.sidebar.expander("📥 دانلود داده آنلاین از Yahoo Finance"):
    st.markdown("""
    <div dir="rtl" style="text-align: right;">
    <b>راهنما:</b>
    <br>نمادها را با کاما و بدون فاصله وارد کنید (مثال: <span style="direction:ltr;display:inline-block">BTC-USD,AAPL,ETH-USD</span>)
    </div>
    """, unsafe_allow_html=True)
    tickers_input = st.text_input("نماد دارایی‌ها (با کاما و بدون فاصله)")
    start = st.date_input("تاریخ شروع", value=pd.to_datetime("2023-01-01"))
    end = st.date_input("تاریخ پایان", value=pd.to_datetime("today"))
    download_btn = st.button("دریافت داده آنلاین")

if download_btn and tickers_input.strip():
    tickers = [t.strip() for t in tickers_input.strip().split(",") if t.strip()]
    try:
        data = yf.download(tickers, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)
        if data.empty:
            st.error("داده‌ای دریافت نشد!")
        else:
            new_downloaded = []
            for t in tickers:
                df, err = get_price_dataframe_from_yf(data, t)
                if df is not None:
                    df['Date'] = pd.to_datetime(df['Date'])
                    new_downloaded.append((t, df))
                    st.success(f"داده {t} با موفقیت دانلود شد.")
                else:
                    st.error(f"{err}")
            st.session_state["downloaded_dfs"].extend(new_downloaded)
    except Exception as ex:
        st.error(f"خطا در دریافت داده: {ex}")

period = st.sidebar.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'هفتگی'])
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'هفتگی': 'W'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'هفتگی': 52}[period]

st.sidebar.markdown("---")
user_rf = st.sidebar.number_input("نرخ بدون ریسک سالانه (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.1) / 100

st.sidebar.markdown("---")
investment_amount = st.sidebar.number_input("💵 سرمایه کل (دلار)", min_value=0.0, value=float(st.session_state["investment_amount"]), step=100.0)
st.session_state["investment_amount"] = investment_amount

# ---------- حداقل و حداکثر وزن دارایی ها ----------
min_weights = []
max_weights = []
asset_names = []
if st.session_state["downloaded_dfs"] or st.session_state["uploaded_dfs"]:
    name_counter = Counter()
    for t, df in st.session_state["downloaded_dfs"] + st.session_state["uploaded_dfs"]:
        base_name = t
        name_counter[base_name] += 1
        name = base_name if name_counter[base_name] == 1 else f"{base_name} ({name_counter[base_name]})"
        asset_names.append(name)

    st.sidebar.markdown("### 🔒 محدودیت وزن هر دارایی در پرتفو")
    cols = st.sidebar.columns(2)
    for i, name in enumerate(asset_names):
        with cols[i%2]:
            min_w = st.number_input(f"حداقل وزن {name}", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key=f"minw_{name}")
            max_w = st.number_input(f"حداکثر وزن {name}", min_value=0.0, max_value=1.0, value=1.0, step=0.01, key=f"maxw_{name}")
            min_weights.append(min_w)
            max_weights.append(max_w)
    min_weights = np.array(min_weights)
    max_weights = np.array(max_weights)

# ---------- Main Analysis ----------
if st.session_state["downloaded_dfs"] or st.session_state["uploaded_dfs"]:
    # 1- آماده‌سازی دیتافریم قیمتی
    name_counter = Counter()
    df_list = []
    asset_names = []
    for t, df in st.session_state["downloaded_dfs"] + st.session_state["uploaded_dfs"]:
        base_name = t
        name_counter[base_name] += 1
        name = base_name if name_counter[base_name] == 1 else f"{base_name} ({name_counter[base_name]})"
        temp_df = df.copy()
        temp_df = temp_df.rename(columns={"Price": name})
        temp_df = temp_df.dropna(subset=[name])
        temp_df = temp_df.set_index("Date")
        asset_names.append(name)
        df_list.append(temp_df[[name]])
    prices_df = pd.concat(df_list, axis=1, join="inner")
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        prices_df.index = pd.to_datetime(prices_df.index)
    resampled_prices = prices_df.resample(resample_rule).last().dropna()

    # 2- تعریف معاملات آپشن برای هر دارایی
    st.markdown("## ⚙️ تنظیمات معاملات آپشن و بیمه برای هر دارایی")
    st.markdown("برای هر دارایی می‌توانید ترکیبی از خرید/فروش دارایی و اختیار خرید/فروش (کال/پوت) تعریف کنید.")
    st.info("برای بیمه کلاسیک، کافیست 'خرید دارایی' (با حجم مناسب) و 'خرید پوت' (با قیمت اعمال، پریمیوم و حجم مناسب) وارد کنید.")
    option_rows_dict = {}
    for name in asset_names:
        with st.expander(f"⚙️ معاملات {name}", expanded=True):
            opt_rows = []
            for i in range(3): # تا ۳ ردیف آپشن
                c1, c2, c3, c4 = st.columns([2,2,2,2])
                with c1:
                    row_type = st.selectbox("نوع معامله", ['-', 'خرید دارایی', 'فروش دارایی', 'خرید کال', 'فروش کال', 'خرید پوت', 'فروش پوت'],
                                            key=f"opttype_{name}_{i}")
                with c2:
                    strike = st.number_input("قیمت اعمال (در صورت نیاز)", key=f"strike_{name}_{i}")
                with c3:
                    premium = st.number_input("پریمیوم ($)", key=f"premium_{name}_{i}")
                with c4:
                    qty = st.number_input("حجم قرارداد/دارایی", key=f"qty_{name}_{i}")
                if row_type != '-' and qty != 0:
                    opt_rows.append((row_type, strike, premium, qty))
            option_rows_dict[name] = opt_rows

    st.session_state["option_rows"] = option_rows_dict.copy()

    # 3- ساخت سری زمانی بازده آپشن + دارایی پایه
    returns_dict = {}
    for name in asset_names:
        price = resampled_prices[name]
        opt_rows = option_rows_dict.get(name, [])
        if opt_rows:
            ret_option = calc_options_series(opt_rows, price)
            returns_dict[name] = ret_option
        else:
            returns_dict[name] = price.pct_change().fillna(0)

    returns_df = pd.DataFrame(returns_dict).dropna()

    # 4- تحلیل پرتفو (شبیه‌سازی با محدودیت وزن)
    mean_returns = returns_df.mean() * annual_factor
    cov_matrix = returns_df.cov() * annual_factor

    n_portfolios = 2500
    all_risks, all_returns, all_weights, all_sharpes, all_cvars = [], [], [], [], []
    cvar_alpha = 0.95

    for i in range(n_portfolios):
        valid = False
        for _ in range(100):
            ws = np.random.dirichlet(np.ones(len(asset_names)), size=1)[0]
            if np.all(ws >= min_weights) and np.all(ws <= max_weights):
                valid = True
                break
        if not valid:
            continue
        port_return = np.dot(ws, mean_returns)
        port_std = np.sqrt(np.dot(ws.T, np.dot(cov_matrix, ws)))
        port_series = returns_df @ ws
        sharpe = (port_return - user_rf) / port_std if port_std > 0 else 0
        # CVaR
        sim_returns = port_series
        var = np.percentile(sim_returns, (1-cvar_alpha)*100)
        cvar = sim_returns[sim_returns <= var].mean() if np.any(sim_returns <= var) else var
        all_risks.append(port_std)
        all_returns.append(port_return)
        all_weights.append(ws)
        all_sharpes.append(sharpe)
        all_cvars.append(-cvar)
    all_risks = np.array(all_risks)
    all_returns = np.array(all_returns)
    all_weights = np.array(all_weights)
    all_sharpes = np.array(all_sharpes)
    all_cvars = np.array(all_cvars)

    # مرز کارا (با محدودیت وزن)
    ef_results, ef_weight_arr = efficient_frontier(mean_returns, cov_matrix, points=300, min_weights=min_weights, max_weights=max_weights)

    # 5- پیدا کردن پرتفو بهینه‌ها
    max_sharpe_idx = np.argmax(all_sharpes)
    min_risk_idx = np.argmin(all_risks)
    max_return_idx = np.argmax(all_returns)
    best_cvar_idx = np.argmin(all_cvars)

    bests = [
        ("بهینه شارپ", all_weights[max_sharpe_idx], all_risks[max_sharpe_idx], all_returns[max_sharpe_idx], "red"),
        ("کم‌ریسک‌ترین", all_weights[min_risk_idx], all_risks[min_risk_idx], all_returns[min_risk_idx], "blue"),
        ("پر بازده‌ترین", all_weights[max_return_idx], all_risks[max_return_idx], all_returns[max_return_idx], "green"),
    ]


    # 6- نمودار مرز کارا حرفه‌ای + خط بازار سرمایه و Sharpe
    st.markdown("## 📊 مرز کارا و تحلیل پرتفو (Sharpe Ratio)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=all_risks*100, y=all_returns*100,
        mode='markers',
        marker=dict(
            color=all_sharpes, colorscale='Viridis', colorbar=dict(title='SharpeRatio'),
            size=7, line=dict(width=0)
        ),
        name='Portfolios',
        hovertemplate='ریسک: %{x:.2f}٪<br>بازده: %{y:.2f}٪<br>Sharpe: %{marker.color:.2f}<extra></extra>'
    ))
    max_risk = all_risks.max() * 1.3 * 100
    sharpe_star = all_sharpes[max_sharpe_idx]
    cal_x = np.linspace(0, max_risk, 100)
    cal_y = user_rf*100 + sharpe_star * cal_x
    fig.add_trace(go.Scatter(
        x=cal_x, y=cal_y, mode='lines',
        line=dict(dash='dash', color='red'), name='خط بازار سرمایه (CAL)'
    ))
    fig.add_trace(go.Scatter(
        x=[all_risks[max_sharpe_idx]*100], y=[all_returns[max_sharpe_idx]*100],
        mode='markers+text', marker=dict(size=14, color='red'),
        text=["بهینه"], textposition="top right", name="پرتفوی بهینه"
    ))
    fig.update_layout(
        title="مرزکارا با رنگ‌بندی Sharpe Ratio",
        xaxis_title="ریسک (%)",
        yaxis_title="بازده (%)",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- نمودار CVaR
    st.markdown(f"## CVaR نمودار ریسک-بازده پرتفوی‌ها با رنگ ({int(cvar_alpha*100)}%)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=all_risks*100, y=all_returns*100,
        mode='markers',
        marker=dict(
            color=all_cvars, colorscale='Inferno', colorbar=dict(title='-CVaR'),
            size=7, line=dict(width=0)
        ),
        name='CVaR Portfolios',
        hovertemplate='ریسک: %{x:.2f}٪<br>بازده: %{y:.2f}٪<br>-CVaR: %{marker.color:.2f}<extra></extra>'
    ))
    fig2.add_trace(go.Scatter(
        x=[all_risks[best_cvar_idx]*100], y=[all_returns[best_cvar_idx]*100],
        mode='markers+text', marker=dict(size=14, color='lime'),
        text=["بهینه CVaR"], textposition="bottom right", name="پرتفوی بهینه CVaR"
    ))
    fig2.update_layout(
        title=f"نمودار CVaR ({int(cvar_alpha*100)}%) ریسک-بازده پرتفوی‌ها با رنگ",
        xaxis_title="ریسک (%)",
        yaxis_title="بازده (%)",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 7- نمایش ترکیب پرتفو بهینه و ابزار تعاملی PnL نقطه‌ای برای هر دارایی
    st.markdown("## 🔎 ترکیب و سود/زیان نقطه‌ای هر دارایی")
    for name in asset_names:
        opt_rows = option_rows_dict.get(name, [])
        st.markdown(f"### {name}")
        if opt_rows:
            with st.expander("نمایش نمودار سود/زیان نقطه‌ای این استراتژی (PnL Option)", expanded=False):
                asset_price = resampled_prices[name].iloc[-1]
                display_price = st.number_input(f"قیمت دارایی در سررسید ({name})", value=float(asset_price), key=f"display_price_{name}")
                price_range = np.linspace(asset_price * 0.7, asset_price * 1.3, 500)
                total_pnl = np.zeros_like(price_range)
                def calculate_pnl(row_type, strike, premium, qty, price_range, asset_price):
                    pnl = np.zeros_like(price_range)
                    if row_type == 'خرید دارایی':
                        pnl = (price_range - asset_price) * qty
                    elif row_type == 'فروش دارایی':
                        pnl = (asset_price - price_range) * qty
                    elif row_type == 'خرید کال':
                        pnl = np.maximum(price_range - strike, 0) * qty - premium * qty
                    elif row_type == 'فروش کال':
                        pnl = -np.maximum(price_range - strike, 0) * qty + premium * qty
                    elif row_type == 'خرید پوت':
                        pnl = np.maximum(strike - price_range, 0) * qty - premium * qty
                    elif row_type == 'فروش پوت':
                        pnl = -np.maximum(strike - price_range, 0) * qty + premium * qty
                    return pnl
                for row in opt_rows:
                    total_pnl += calculate_pnl(*row, price_range, asset_price)
                profit_mask = total_pnl >= 0
                loss_mask = total_pnl < 0
                exact_pnl = np.interp(display_price, price_range, total_pnl)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=price_range[profit_mask], y=total_pnl[profit_mask],
                                          fill='tozeroy', name='سود', line=dict(color='green')))
                fig2.add_trace(go.Scatter(x=price_range[loss_mask], y=total_pnl[loss_mask],
                                          fill='tozeroy', name='زیان', line=dict(color='red')))
                fig2.add_trace(go.Scatter(x=[display_price], y=[exact_pnl],
                                          mode='markers+text', text=[f"{exact_pnl:.2f} $"],
                                          textposition="top center", marker=dict(size=10, color='blue'),
                                          name='قیمت انتخابی'))
                fig2.add_hline(y=0, line_dash='dash', line_color='gray')
                fig2.update_layout(title=f"نمودار سود / زیان استراتژی ({name})",
                                   xaxis_title="قیمت پایانی دارایی ($)",
                                   yaxis_title="سود / زیان ($)",
                                   template="plotly_white", height=370)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("برای این دارایی معامله‌ای تعریف نشده است (صرفاً بازده دارایی پایه لحاظ می‌شود).")

    # 8- نمایش وزن دلاری و دایره ای پرتفو بهینه (شارپ)
    st.markdown("## 💰 وزن دلاری پرتفو بهینه (شارپ)")
    weights = bests[0][1]
    dollar_vals = weights * st.session_state["investment_amount"]
    cols = st.columns(len(asset_names))
    for i, name in enumerate(asset_names):
        percent = weights[i]
        dollar = dollar_vals[i]
        with cols[i]:
            st.markdown(f"""
            <div style='text-align:center;direction:rtl'>
            <b>{name}</b><br>
            {format_percent(percent)}<br>
            {format_money(dollar)}
            </div>
            """, unsafe_allow_html=True)
    # نمودار دایره ای توزیع دلاری پرتفو
    figpie = px.pie(
        values=dollar_vals,
        names=asset_names,
        title="توزیع دلاری پرتفو بهینه (شارپ)",
        hole=0.4
    )
    st.plotly_chart(figpie, use_container_width=True)

else:
    st.warning("⚠️ لطفاً فایل‌های CSV شامل ستون‌های Date و Price یا Close یا Open را آپلود کنید یا از بخش دانلود آنلاین داده استفاده کنید.")
