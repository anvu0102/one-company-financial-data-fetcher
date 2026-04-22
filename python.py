import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import warnings
import io

# --- 1. CẤU HÌNH ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Financial Command Center", layout="wide", page_icon="💎")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# --- 2. HÀM TẢI DỮ LIỆU & TÍNH TOÁN ---
@st.cache_resource(show_spinner=False)
def get_full_stock_data(symbol):
    ticker_symbol = symbol.strip().upper()
    us_indices = ['AAPL', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'BTC-USD', 'NFLX']
    if len(ticker_symbol) == 3 and ticker_symbol not in us_indices: 
        ticker_symbol = f"{ticker_symbol}.VN"
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
        if hist.empty: return None
        return {
            "stock": stock, "hist": hist, "info": stock.info,
            "financials_y": stock.financials, "balance_sheet_y": stock.balance_sheet,
            "financials_q": stock.quarterly_financials, "balance_sheet_q": stock.quarterly_balance_sheet
        }
    except: return None

def calculate_technical_all(df):
    df = df.copy()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/(loss + 1e-9)))
    # MA & Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (df['STD20'] * 2)
    df['BB_Lower'] = df['MA20'] - (df['STD20'] * 2)
    return df

def monte_carlo_simulation(df, days=30, simulations=100):
    returns = df['Close'].pct_change().dropna()
    last_price = df['Close'].iloc[-1]
    results = np.zeros((days, simulations))
    for i in range(simulations):
        prices = [last_price]
        for _ in range(days - 1):
            prices.append(prices[-1] * (1 + np.random.normal(returns.mean(), returns.std())))
        results[:, i] = prices
    return results

# --- 3. TRÍ TUỆ NHÂN TẠO (GEMINI-2.5-FLASH-LITE) ---
def analyze_with_ai(symbol, data_dict, info, tech_data):
    if not GEMINI_API_KEY: return "⚠️ API Key không khả dụng."
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        context = (
            f"Mã: {symbol}. Giá: {info.get('currentPrice', 0):,.0f}. RSI: {tech_data['RSI'].iloc[-1]:.2f}\n"
            f"MA20: {tech_data['MA20'].iloc[-1]:.0f}, MA50: {tech_data['MA50'].iloc[-1]:.0f}\n"
            f"Biên lợi nhuận: {info.get('grossMargins', 0)*100:.1f}%\n"
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=[f"Phân tích kỹ thuật & cơ bản toàn diện cho: {context}. Đưa ra khuyến nghị ngắn gọn."]
        )
        return response.text
    except Exception as e: return f"❌ Lỗi AI: {str(e)}"

# --- 4. GIAO DIỆN CHÍNH ---
st.title("💎 AI Financial Command Center")

symbol = st.sidebar.text_input("Nhập mã chứng khoán:", value="AAPL").upper()
period_choice = st.sidebar.radio("Kỳ báo cáo:", ['Năm', 'Quý'])
suffix = '_y' if period_choice == 'Năm' else '_q'

if symbol:
    data = get_full_stock_data(symbol)
    if data:
        info, hist = data['info'], data['hist']
        df_plot = calculate_technical_all(hist)
        
        # --- HEADER METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Giá", f"{info.get('currentPrice', hist['Close'].iloc[-1]):,.0f}", f"{info.get('trailingPE', 0):.2f} P/E")
        m2.metric("Vốn hóa", f"{info.get('marketCap', 0)/1e12:.2f}T")
        m3.metric("RSI (14)", f"{df_plot['RSI'].iloc[-1]:.1f}")
        m4.metric("Dự báo mục tiêu", f"{info.get('targetMeanPrice', 0):,.0f}")

        tabs = st.tabs(["📉 Đồ thị Chuyên sâu", "🔮 Dự báo Monte Carlo", "🧠 Phân tích AI", "📊 Tài chính & Tin tức"])

        # TAB 1: BIỂU ĐỒ CHUYÊN SÂU
        with tabs[0]:
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
                               row_heights=[0.6, 0.2, 0.2], subplot_titles=("Giá & Bollinger Bands", "Volume", "RSI"))
            # Price & BB
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_Upper'], name="BB Upper", line=dict(color='rgba(173,216,230,0.4)', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_Lower'], name="BB Lower", line=dict(color='rgba(173,216,230,0.4)', dash='dash'), fill='tonexty'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], name="MA50", line=dict(color='#2196F3', width=2)), row=1, col=1)
            # Volume
            colors = ['red' if df_plot['Open'].iloc[i] > df_plot['Close'].iloc[i] else 'green' for i in range(len(df_plot))]
            fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], name="Khối lượng", marker_color=colors), row=2, col=1)
            # RSI
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name="RSI", line=dict(color='#9C27B0')), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        # TAB 2: DỰ BÁO MONTE CARLO
        with tabs[1]:
            st.subheader(f"Dự báo biến động giá {symbol} trong 30 ngày tới")
            sim_results = monte_carlo_simulation(hist, days=30, simulations=100)
            fig_sim = go.Figure()
            for i in range(100):
                fig_sim.add_trace(go.Scatter(y=sim_results[:, i], mode='lines', line=dict(width=0.5), opacity=0.3, showlegend=False))
            fig_sim.update_layout(title="Giả lập 100 kịch bản giá", xaxis_title="Ngày", yaxis_title="Giá dự kiến", template="plotly_dark")
            st.plotly_chart(fig_sim, use_container_width=True)
            st.info("⚠️ Đây là mô hình giả lập dựa trên độ lệch chuẩn quá khứ, chỉ mang tính chất tham khảo.")

        # TAB 3: AI Analysis
        with tabs[2]:
            if st.button("🚀 Kích hoạt Gemini 2.5 Flash-Lite"):
                current_fin = { 'financials': data[f'financials{suffix}'] }
                with st.spinner("Đang xử lý thuật toán..."):
                    res = analyze_with_ai(symbol, current_fin, info, df_plot)
                    st.markdown(res)

        # TAB 4: FINANCIAL & NEWS
        with tabs[3]:
            col_l, col_r = st.columns([1, 2])
            with col_l:
                st.subheader("📊 Tài chính")
                st.dataframe(data[f"financials{suffix}"].iloc[:, :4])
                # Download Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    data[f"financials{suffix}"].to_excel(writer, sheet_name="Report")
                st.download_button("📥 Tải báo cáo", output.getvalue(), f"{symbol}_data.xlsx")
            with col_r:
                st.subheader("📰 Tin tức mới nhất")
                try:
                    news = data['stock'].news
                    if news:
                        for item in news[:6]:
                            with st.container():
                                st.markdown(f"**[{item.get('title', 'N/A')}]({item.get('link', '#')})**")
                                st.caption(f"Nguồn: {item.get('publisher', 'Yahoo Finance')}")
                                st.divider()
                    else: st.info("Không có tin tức mới.")
                except: st.error("Lỗi tải tin tức.")
    else:
        st.error("⚠️ Không tìm thấy mã này. Hãy thử lại (VD: VNM, AAPL, FPT).")
