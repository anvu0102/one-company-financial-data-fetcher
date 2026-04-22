import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
import io

# --- 1. CẤU HÌNH ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Terminal Pro 2026", layout="wide", page_icon="⚡")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# --- 2. HÀM TẢI & XỬ LÝ DỮ LIỆU ---
@st.cache_resource(show_spinner=False)
def get_full_stock_data(symbol):
    ticker_symbol = symbol.strip().upper()
    us_indices = ['AAPL', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'BTC-USD']
    if len(ticker_symbol) == 3 and ticker_symbol not in us_indices: 
        ticker_symbol = f"{ticker_symbol}.VN"
    
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="2y") # Lấy 2 năm để có dữ liệu dự báo
        if hist.empty: return None
        return {
            "stock": stock, "hist": hist, "info": stock.info,
            "financials_y": stock.financials, "balance_sheet_y": stock.balance_sheet, "cashflow_y": stock.cashflow,
            "financials_q": stock.quarterly_financials, "balance_sheet_q": stock.quarterly_balance_sheet, "cashflow_q": stock.quarterly_cashflow
        }
    except: return None

def calculate_advanced_metrics(df):
    df = df.copy()
    # Kỹ thuật cơ bản
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/(loss + 1e-9)))
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Tính dự báo xu hướng (Linear Regression)
    df['Time_Index'] = np.arange(len(df))
    X = df[['Time_Index']].values[-60:] # Lấy 60 phiên gần nhất
    y = df['Close'].values[-60:]
    model = LinearRegression().fit(X, y)
    df['Trend_Line'] = model.predict(df[['Time_Index']].values)
    return df

# --- 3. AI ANALYSIS (GEMINI-2.5-FLASH-LITE) ---
def analyze_with_ai(symbol, data_dict, info, tech_data):
    if not GEMINI_API_KEY: return "⚠️ Thiếu API Key trong Secrets."
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        context = (
            f"Mã: {symbol}. Giá: {info.get('currentPrice'):,.0f}. RSI: {tech_data['RSI'].iloc[-1]:.2f}\n"
            f"P/E: {info.get('trailingPE')}. ROE: {info.get('returnOnEquity')}\n"
            f"Tài chính gần nhất:\n{data_dict['financials'].iloc[:, :2].to_string()}"
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=[f"Phân tích đa chiều (Kỹ thuật + Cơ bản) và cho điểm đầu tư (1-10): {context}"]
        )
        return response.text
    except Exception as e: return f"❌ Lỗi AI: {str(e)}"

# --- 4. GIAO DIỆN CHÍNH ---
st.title("⚡ AI Stock Terminal Pro")

symbol = st.sidebar.text_input("Mã cổ phiếu:", value="AAPL").upper()
period_choice = st.sidebar.radio("Kỳ báo cáo:", ['Năm', 'Quý'])
suffix = '_y' if period_choice == 'Năm' else '_q'

if symbol:
    data = get_full_stock_data(symbol)
    if data:
        info = data['info']
        hist = data['hist']
        df_plot = calculate_advanced_metrics(hist)
        
        # --- TOP METRICS ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Giá hiện tại", f"{info.get('currentPrice', hist['Close'].iloc[-1]):,.0f}")
        c2.metric("Vốn hóa", f"{info.get('marketCap', 0)/1e12:.2f}T")
        c3.metric("P/E Trailing", f"{info.get('trailingPE', 0):.2f}")
        c4.metric("Dự báo xu hướng", "Tăng" if df_plot['Trend_Line'].iloc[-1] > df_plot['Trend_Line'].iloc[-10] else "Giảm")

        tabs = st.tabs(["📈 Kỹ thuật & Dự báo", "🧠 AI Strategic Insight", "📊 Sức khỏe tài chính", "📰 Tin tức"])

        # TAB 1: BIỂU ĐỒ NÂNG CAO
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
            # Candle
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Giá"), row=1, col=1)
            # MA & Trend
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], name="MA20", line=dict(color='orange', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], name="MA50", line=dict(color='blue', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index[-60:], y=df_plot['Trend_Line'].values[-60:], name="Dự báo hồi quy", line=dict(color='red', dash='dot')), row=1, col=1)
            # RSI
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        # TAB 2: AI INSIGHT
        with tabs[1]:
            if st.button("🚀 Kích hoạt AI Analysis"):
                with st.spinner("Gemini 2.5 Flash-Lite đang tổng hợp dữ liệu..."):
                    res = analyze_with_ai(symbol, {"financials": data[f"financials{suffix}"]}, info, df_plot)
                    st.markdown(res)

        # TAB 3: FINANCIAL HEALTH (TÍNH NĂNG MỚI)
        with tabs[2]:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Chỉ số hiệu quả")
                fin_y = data['financials_y']
                bs_y = data['balance_sheet_y']
                try:
                    # Tính toán thủ công một số tỷ lệ
                    current_assets = bs_y.loc['Total Current Assets'].iloc[0]
                    current_liab = bs_y.loc['Total Current Liabilities'].iloc[0]
                    quick_ratio = current_assets / current_liab
                    st.write(f"**Tỷ số thanh toán hiện hành:** {quick_ratio:.2f}")
                    st.progress(min(quick_ratio/3, 1.0)) # Thanh đo trực quan
                    
                    st.write(f"**ROE:** {info.get('returnOnEquity', 0)*100:.2f}%")
                    st.write(f"**Biên lợi nhuận ròng:** {info.get('netIncomeToCommon', 0)/info.get('totalRevenue', 1)*100:.2f}%")
                except: st.info("Dữ liệu chi tiết đang được cập nhật.")

            with col2:
                st.subheader("Cơ cấu Dòng tiền")
                cf_y = data['cashflow_y']
                try:
                    cf_data = pd.DataFrame({
                        'Loại': ['Kinh doanh', 'Đầu tư', 'Tài chính'],
                        'Giá trị': [cf_y.loc['Operating Cash Flow'].iloc[0], 
                                   cf_y.loc['Investing Cash Flow'].iloc[0], 
                                   cf_y.loc['Financing Cash Flow'].iloc[0]]
                    })
                    fig_cf = go.Figure(data=[go.Bar(x=cf_data['Loại'], y=cf_data['Giá trị'], marker_color=['green', 'red', 'blue'])])
                    fig_cf.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig_cf, use_container_width=True)
                except: st.write("Không đủ dữ liệu dòng tiền.")

        # TAB 4: TIN TỨC
        with tabs[3]:
            news = data['stock'].news
            if news:
                for item in news[:8]:
                    with st.container():
                        title = item.get('title', 'No Title')
                        st.markdown(f"**[{title}]({item.get('link', '#')})**")
                        st.caption(f"Nguồn: {item.get('publisher')} | Loại: {item.get('type')}")
                        st.divider()
            else: st.info("Không có tin tức mới.")
    else:
        st.error("⚠️ Không tìm thấy dữ liệu.")
