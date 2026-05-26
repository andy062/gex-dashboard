import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="Hedge Fund Quant Dashboard - Multi-Horizon", layout="wide")
st.title("🏛️ Institutional Risk & Liquidity Monitor")
st.markdown("Analisi quantitativa multi-orizzonte: Gamma Exposure, Vanna Exposure e Volatility Skew.")

def calculate_greeks(S, K, t, r, sigma):
    if t <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0, 0.0
    d1 = (np.log(S / K) + (r + (sigma ** 2) / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(t))
    vanna = -norm.pdf(d1) * d2 / sigma
    return gamma, vanna

st.sidebar.header("🛠️ Parametri di Input")
ticker = st.sidebar.text_input("Asset Ticker (USA)", value="AAPL").upper()
horizon = st.sidebar.selectbox("📆 Orizzonte Temporale Opzioni", options=["Breve Termine (3 scadenze)", "Medio Termine (6 scadenze)", "Lungo Termine (Tutte)"], index=0)
risk_free_rate = st.sidebar.number_input("Tasso Risk-Free (r)", value=0.045, step=0.005, format="%.3f")
vix_proxy = st.sidebar.number_input("Volatilità Implicita Base (σ)", value=0.20, step=0.01, format="%.2f")

if "Breve" in horizon: max_expirations = 3
elif "Medio" in horizon: max_expirations = 6
else: max_expirations = 999

@st.cache_data(ttl=300)
def fetch_and_process_advanced_risk(ticker, r, sigma, max_exp):
    stock = yf.Ticker(ticker)
    try: spot_price = stock.history(period="1d")["Close"].iloc[-1]
    except Exception: return None, None, None
    dates = stock.options
    if not dates: return None, spot_price, None
    num_expirations = min(max_exp, len(dates))
    aggregated_data = {}
    for i in range(num_expirations):
        target_date = dates[i]
        try:
            opt_chain = stock.option_chain(target_date)
            calls = opt_chain.calls[['strike', 'openInterest', 'impliedVolatility']].dropna()
            puts = opt_chain.puts[['strike', 'openInterest', 'impliedVolatility']].dropna()
            today = pd.Timestamp.now()
            exp_date = pd.Timestamp(target_date)
            days_to_exp = max(1, (exp_date - today).days)
            t = days_to_exp / 365.0
            chain = pd.merge(calls, puts, on='strike', suffixes=('_call', '_put'), how='outer').fillna(0)
            for _, row in chain.iterrows():
                strike = row['strike']
                if strike < spot_price * 0.75 or strike > spot_price * 1.25: continue
                oi_call = row['openInterest_call']
                oi_put = row['openInterest_put']
                iv_call = row['impliedVolatility_call'] if row['impliedVolatility_call'] > 0 else sigma
                iv_put = row['impliedVolatility_put'] if row['impliedVolatility_put'] > 0 else sigma
                gamma_c, vanna_c = calculate_greeks(spot_price, strike, t, r, iv_call)
                gamma_p, vanna_p = calculate_greeks(spot_price, strike, t, r, iv_put)
                call_gex = oi_call * gamma_c * 100 * spot_price * 0.01
                put_gex = oi_put * gamma_p * 100 * spot_price * 0.01 * (-1)
                total_gex = call_gex + put_gex
                call_vex = oi_call * vanna_c * 100 * 0.01
                put_vex = oi_put * vanna_p * 100 * 0.01 * (-1)
                total_vex = call_vex + put_vex
                skew = iv_put - iv_call if (row['impliedVolatility_put'] > 0 and row['impliedVolatility_call'] > 0) else 0.0
                if strike not in aggregated_data: aggregated_data[strike] = {"Total GEX ($)": 0.0, "Total VEX ($)": 0.0, "Skew (IV Diff)": [], "Count": 0}
                aggregated_data[strike]["Total GEX ($)"] += total_gex
                aggregated_data[strike]["Total VEX ($)"] += total_vex
                if skew != 0: aggregated_data[strike]["Skew (IV Diff)"].append(skew)
                aggregated_data[strike]["Count"] += 1
        except Exception: continue
    if not aggregated_data: return None, spot_price, None
    final_rows = []
    for strike, values in aggregated_data.items():
        avg_skew = np.mean(values["Skew (IV Diff)"]) if values["Skew (IV Diff)"] else 0.0
        final_rows.append({"Strike": strike, "Total GEX ($)": values["Total GEX ($)"], "Total VEX ($)": values["Total VEX ($)"], "Skew (IV Diff)": avg_skew})
    df_risk = pd.DataFrame(final_rows).sort_values(by="Strike")
    return df_risk, spot, f"Aggregazione di {num_expirations} scadenze opzioni ({horizon.split(' ')[0]} Termine)"

df_risk, spot, info_scadenza = fetch_and_process_advanced_risk(ticker, risk_free_rate, vix_proxy, max_expirations)

if spot is None: st.error(f"Impossibile trovare l'asset '{ticker}'.")
elif df_risk is None or df_risk.empty: st.warning(f"Nessun dato valido trovato per il ticker {ticker} con questo orizzonte temporale.")
else:
    net_gex = df_risk["Total GEX ($)"].sum()
    net_vex = df_risk["Total VEX ($)"].sum()
    avg_skew = df_risk["Skew (IV Diff)"].mean() * 100
    negative_gex = df_risk[df_risk["Total GEX ($)"] < 0]
    flip_zone = negative_gex["Strike"].max() if not negative_gex.empty else df_risk["Strike"].min()
    if flip_zone == df_risk["Strike"].min() or flip_zone == df_risk["Strike"].max():
        flip_zone = df_risk.iloc[(df_risk['Total GEX ($)'].abs()).argsort()[:1]]['Strike'].values[0]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Prezzo Spot", f"${spot:.2f}")
    col2.metric("Net GEX (Gamma)", f"${net_gex:,.2f}", delta="Long Gamma (Stabile)" if net_gex > 0 else "Short Gamma (Volatile)", delta_color="normal" if net_gex > 0 else "inverse")
    col3.metric("Stima Gamma Flip Zone", f"${flip_zone:.2f}")
    col4.metric("Net VEX (Vanna)", f"${net_vex:,.2f}", delta="Vanna di Sostegno" if net_vex > 0 else "Rischio Accelerazione", delta_color="normal" if net_vex > 0 else "inverse")
    col5.metric("Avg Volatility Skew", f"{avg_skew:.2f}%")
    st.info(f"Stato del Data Engine: **{info_scadenza}**")
    tab1, tab2, tab3 = st.tabs(["📊 Gamma Exposure (GEX)", "🌊 Vanna Exposure (VEX)", "📈 Volatility Skew Curve"])
    with tab1:
        fig_gex = go.Figure()
        fig_gex.add_trace(go.Bar(x=df_risk["Strike"], y=df_risk["Total GEX ($)"], name="Net GEX", marker_color=np.where(df_risk["Total GEX ($)"] > 0, '#00ffbb', '#ff0055')))
        fig_gex.add_vline(x=spot, line_width=2, line_dash="dash", line_color="white", annotation_text="Prezzo Spot")
        fig_gex.add_vline(x=flip_zone, line_width=1.5, line_dash="dot", line_color="#ffc107", annotation_text=f"Flip Zone: ${flip_zone:.2f}")
        fig_gex.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500, xaxis_title="Strike Price ($)", yaxis_title="GEX ($)")
        st.plotly_chart(fig_gex, use_container_width=True)
    with tab2:
        fig_vex = go.Figure()
        fig_vex.add_trace(go.Bar(x=df_risk["Strike"], y=df_risk["Total VEX ($)"], name="Net VEX", marker_color=np.where(df_risk["Total VEX ($)"] > 0, '#00e5ff', '#ff6d00')))
        fig_vex.add_vline(x=spot, line_width=2, line_dash="dash", line_color="white", annotation_text="Prezzo Spot")
        fig_vex.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500, xaxis_title="Strike Price ($)", yaxis_title="VEX ($)")
        st.plotly_chart(fig_vex, use_container_width=True)
    with tab3:
        fig_skew = go.Figure()
        fig_skew.add_trace(go.Scatter(x=df_risk["Strike"], y=df_risk["Skew (IV Diff)"]*100, mode='lines+markers', line=dict(color='#d500f9', width=2)))
        fig_skew.add_vline(x=spot, line_width=2, line_dash="dash", line_color="white", annotation_text="Prezzo Spot")
        fig_skew.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500, xaxis_title="Strike Price ($)", yaxis_title="Skew Margin (Points)")
        st.plotly_chart(fig_skew, use_container_width=True)
    with st.expander("🔍 Tabella Dati Quantitativi Integrata"):
        st.dataframe(df_risk.style.format({"Strike": "${:.2f}", "Total GEX ($)": "${:,.2f}", "Total VEX ($)": "${:,.2f}", "Skew (IV Diff)": "{:.4f}"}), use_container_width=True)
