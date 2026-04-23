import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
import warnings
import io

# --- 1. CẤU HÌNH HỆ THỐNG ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Stock Terminal Pro", layout="wide", page_icon="⚡")

# Lấy API Key từ Streamlit Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# --- 2. HÀM TẢI DỮ LIỆU CẢI TIẾN (CHỐNG RATE LIMIT) ---
@st.cache_resource(ttl=3600, show_spinner=False)
def get_full_stock_data(symbol):
    if not symbol: return None
    
    ticker_symbol = symbol.strip().upper()
    global_stocks = ['AAPL', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'BTC-USD', 'ETH-USD', 'GOLD']
    if len(ticker_symbol) == 3 and ticker_symbol not in global_stocks:
        ticker_symbol = f"{ticker_symbol}.VN"
    
    try:
        # Giả lập trình duyệt để tránh bị Yahoo chặn (Rate Limit)
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        })
        
        stock = yf.Ticker(ticker_symbol, session=session)
        hist = stock.history(period="2y")
        
        if hist.empty:
            # Thử lại lần cuối với mã gốc
            stock = yf.Ticker(symbol.strip().upper(), session=session)
            hist = stock.history(period="2y")
            if hist.empty: return None
            
        return {
            "stock": stock, "hist": hist, "info": stock.info,
            "financials_y": stock.financials, "balance_sheet_y": stock.balance_sheet, "cashflow_y": stock.cashflow,
            "financials_q": stock.quarterly_financials, "balance_sheet_q": stock.quarterly_balance_sheet, "cashflow_q": stock.quarterly_cashflow
        }
    except Exception as e:
        if "Too Many Requests" in str(e):
            st.error("⚠️ Yahoo Finance đang hạn chế truy cập (Rate Limit). Vui lòng đợi vài phút.")
        return None

def calculate_indicators(df):
    if df is None or df.empty: return None
    df = df.copy()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/(loss + 1e-9)))
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    # Hồi quy tuyến tính (Dự báo xu hướng 60 phiên)
    df['Idx'] = np.arange(len(df))
    last_60 = df.dropna(subset=['Close']).tail(60)
    if len(last_60) > 10:
        model = LinearRegression().fit(last_60[['Idx']], last_60['Close'])
        df['Trend'] = model.predict(df[['Idx']])
    return df

# --- 3. TRÍ TUỆ NHÂN TẠO (GEMINI-2.5-FLASH-LITE) ---
def analyze_with_ai(symbol, data_dict, info, tech_data):
    if not GEMINI_API_KEY: return "⚠️ Vui lòng cấu hình GEMINI_API_KEY."
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        last_price = info.get('currentPrice', tech_data['Close'].iloc[-1])
        context = (
            f"Mã: {symbol}. Giá: {last_price:,.0f}. RSI: {tech_data['RSI'].iloc[-1]:.2f}\n"
            f"MA20: {tech_data['MA20'].iloc[-1]:.0f}, MA50: {tech_data['MA50'].iloc[-1]:.0f}\n"
            f"Tài chính:\n{data_dict['financials'].iloc[:, :2].to_string()}"
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=[f"Phân tích chuyên sâu kỹ thuật & cơ bản, đưa ra khuyến nghị đầu tư: {context}"]
        )
        return response.text
    except Exception as e: return f"❌ Lỗi AI: {str(e)}"

# --- 4. GIAO DIỆN CHÍNH ---
st.sidebar.header("⚙️ Cấu hình")
user_input = st.sidebar.text_input("Nhập mã cổ phiếu:", value="AAPL")
period_choice = st.sidebar.radio("Kỳ báo cáo:", ['Năm', 'Quý'])
suffix = '_y' if period_choice == 'Năm' else '_q'

if user_input:
    data = get_full_stock_data(user_input)
    
    if data:
        info, hist = data['info'], data['hist']
        df_plot = calculate_indicators(hist)
        
        # --- TOP METRICS ---
        st.subheader(f"🚀 {info.get('longName', user_input.upper())}")
        c1, c2, c3, c4 = st.columns(4)
        price = info.get('currentPrice', hist['Close'].iloc[-1])
        c1.metric("Giá", f"{price:,.0f}", f"{info.get('trailingPE', 0):.2f} PE")
        c2.metric("Vốn hóa", f"{info.get('marketCap', 0)/1e12:.2f}T")
        c3.metric("RSI (14)", f"{df_plot['RSI'].iloc[-1]:.1f}")
        c4.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.1f}%")

        tabs = st.tabs(["📈 Biểu đồ kỹ thuật", "🧠 Phân tích AI", "📊 Tài chính", "📰 Tin tức"])

        # Tab 1: Kỹ thuật (MA20, MA50, Trend, RSI Thresholds)
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.7, 0.3])
            # Nến Nhật
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Giá"), row=1, col=1)
            # MA 20 & 50
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], name="MA20 (Cam)", line=dict(color='orange', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], name="MA50 (Xanh)", line=dict(color='#2196F3', width=1.5)), row=1, col=1)
            # Hồi quy tuyến tính
            if 'Trend' in df_plot:
                fig.add_trace(go.Scatter(x=df_plot.index[-60:], y=df_plot['Trend'].iloc[-60:], name="Dự báo Xu hướng", line=dict(color='red', dash='dot')), row=1, col=1)
            # RSI & Thresholds
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", row=2, col=1)
            
            fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        # Tab 2: AI Strategic Analysis
        with tabs[1]:
            if st.button("🚀 Khởi chạy Gemini 2.5 Flash-Lite"):
                with st.spinner("AI đang 'soi' báo cáo..."):
                    res = analyze_with_ai(user_input, {"financials": data[f"financials{suffix}"]}, info, df_plot)
                    st.info(res)

        # Tab 3: Financial Health
        with tabs[2]:
            st.write("### Sức khỏe tài chính")
            col_l, col_r = st.columns(2)
            with col_l:
                st.dataframe(data[f"balance_sheet{suffix}"], use_container_width=True)
            with col_r:
                try:
                    margin = (info.get('netIncomeToCommon', 0) / info.get('totalRevenue', 1)) * 100
                    st.metric("Biên lợi nhuận ròng", f"{margin:.2f}%")
                    st.metric("Nợ/Vốn CSH", f"{info.get('debtToEquity', 0):.2f}")
                    # Chart dòng tiền nhanh
                    st.bar_chart(data[f"cashflow{suffix}"].iloc[:3, 0])
                except: st.write("Đang tính toán chỉ số...")

        # Tab 4: News Feed
        with tabs[3]:
            try:
                news = data['stock'].news
                if news:
                    for n in news[:8]:
                        title = n.get('title') or n.get('content', {}).get('title', 'No Title')
                        link = n.get('link') or n.get('content', {}).get('clickThroughUrl', {}).get('url', '#')
                        if title != 'No Title':
                            st.markdown(f"**[{title}]({link})**")
                            st.caption(f"Nguồn: {n.get('publisher', 'Yahoo Finance')}")
                            st.divider()
                else: st.info("Không có tin tức mới.")
            except: st.write("Tạm thời không thể tải tin tức.")

    else:
        st.error("❌ Không tìm thấy dữ liệu. Yahoo có thể đang hạn chế yêu cầu của bạn.")
        st.info("Mẹo: FPT -> FPT.VN. Hãy thử lại sau vài phút nếu gặp lỗi 'Too Many Requests'.")
