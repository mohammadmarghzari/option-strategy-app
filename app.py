import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Option Strategy", layout="wide")

st.title("📈 ابزار محاسبه سود و زیان اختیار معامله")

asset_price = st.number_input("قیمت فعلی دارایی پایه ($)", value=1800.0)
display_price = st.number_input("قیمت دارایی در سررسید برای بررسی ($)", value=1800.0)

st.markdown("### تنظیمات معاملات (تا ۵ ردیف)")

rows = []
for i in range(5):
    with st.expander(f"📌 ردیف {i+1}", expanded=(i == 0)):
        row_type = st.selectbox("نوع معامله", ['خرید دارایی', 'فروش دارایی', 'خرید کال', 'فروش کال', 'خرید پوت', 'فروش پوت'], key=f"type_{i}")
        strike = st.number_input("قیمت اعمال (در صورت نیاز)", key=f"strike_{i}")
        premium = st.number_input("پریمیوم ($)", key=f"premium_{i}")
        qty = st.number_input("حجم قرارداد", key=f"qty_{i}")
        rows.append((row_type, strike, premium, qty))

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

# محاسبه سود و زیان کلی
price_range = np.linspace(asset_price * 0.7, asset_price * 1.3, 500)
total_pnl = np.zeros_like(price_range)

for row in rows:
    total_pnl += calculate_pnl(*row, price_range, asset_price)

profit_mask = total_pnl >= 0
loss_mask = total_pnl < 0
exact_pnl = np.interp(display_price, price_range, total_pnl)

fig = go.Figure()
fig.add_trace(go.Scatter(x=price_range[profit_mask], y=total_pnl[profit_mask],
                         fill='tozeroy', name='Profit', line=dict(color='green')))
fig.add_trace(go.Scatter(x=price_range[loss_mask], y=total_pnl[loss_mask],
                         fill='tozeroy', name='Loss', line=dict(color='red')))
fig.add_trace(go.Scatter(x=[display_price], y=[exact_pnl],
                         mode='markers+text', text=[f"{exact_pnl:.2f} $"],
                         textposition="top center", marker=dict(size=10, color='blue'),
                         name='Selected Price'))
fig.add_hline(y=0, line_dash='dash', line_color='gray')
fig.update_layout(title="نمودار سود / زیان استراتژی",
                  xaxis_title="قیمت پایانی دارایی ($)",
                  yaxis_title="سود / زیان ($)",
                  template="plotly_white", height=550)

st.plotly_chart(fig, use_container_width=True)
