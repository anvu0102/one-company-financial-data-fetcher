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

# --- 1. CẤU HÌNH ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Stock Terminal 2026", layout="wide", page_icon="📈")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# --- 2. HÀM TẢI DỮ LIỆU CẢI TIẾN (BỔ SUNG XỬ LÝ LỖI HẠN CHẾ) ---
@st.cache_resource(ttl=3600, show_spinner=False)
def get_full_stock_data(symbol):
    if not symbol: return None
    
    ticker_symbol = symbol.strip().upper()
    global_stocks = ['AAPL', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'BTC-USD', 'ETH-USD', 'GOLD']
    if len(ticker_symbol) == 3 and ticker_symbol not in global_stocks:
        ticker_symbol = f"{ticker_symbol}.VN"
    
    try:
        session = requests.Session()
        # Thay đổi User-Agent liên tục hoặc dùng cái phổ biến để tránh bị chặn
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        })
        
        stock = yf.Ticker(ticker_symbol, session=session)
        
        # Thử tải lịch sử - đây là bước dễ bị Rate Limit nhất
        hist = stock.history(period="2y")
        
        if hist.empty:
            # Dự phòng: Thử lại với mã gốc nếu mã .VN không có dữ liệu
            stock = yf.Ticker(symbol.strip().upper(), session=session)
            hist = stock.history(period="2y")
            if hist.empty: return None
            
        return {
            "stock": stock, "hist": hist, "info": stock.info,
            "financials_y": stock.financials, "balance_sheet_y": stock.balance_sheet, "cashflow_y": stock.cashflow,
            "financials_q": stock.quarterly_financials, "balance_sheet_q": stock.quarterly_balance_sheet, "cashflow_q": stock.quarterly_cashflow
        }
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "Too Many Requests" in error_msg:
            # HIỂN THỊ THÔNG BÁO HƯỚNG DẪN KHI BỊ CHẶN
            st.warning("⚠️ **Hệ thống đang bị tạm khóa (Rate Limit) bởi Yahoo Finance.**")
            st.info("""
            **Cách khắc phục:**
            1. Đợi khoảng 5-10 phút để IP được giải phóng.
            2. Nếu bạn đang dùng VPN, hãy thử tắt/đổi vùng.
            3. Hạn chế nhấn 'Enter' hoặc thay đổi mã quá nhanh liên tục.
            """)
            return "RATE_LIMITED" # Trả về flag đặc biệt
        return None

# --- 3. LOGIC HIỂN THỊ CHÍNH ---
st.sidebar.header("⚙️ Cấu hình")
user_input = st.sidebar.text_input("Nhập mã cổ phiếu:", value="AAPL")

if user_input:
    data = get_full_stock_data(user_input)
    
    if data == "RATE_LIMITED":
        st.error("🚫 Không thể tải dữ liệu mới do giới hạn truy cập từ máy chủ Yahoo.")
        st.stop() # Dừng ứng dụng tại đây để không gây thêm request lỗi
        
    elif data:
        # --- TIẾP TỤC CÁC TÍNH NĂNG HIỂN THỊ NHƯ CŨ ---
        info, hist = data['info'], data['hist']
        
        # Hàm tính toán chỉ báo (giữ nguyên từ bản trước)
        def calculate_indicators(df):
            df = df.copy()
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + gain/(loss + 1e-9)))
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            return df

        df_plot = calculate_indicators(hist)
        
        # Header Metrics
        st.subheader(f"🚀 {info.get('longName', user_input.upper())}")
        c1, c2, c3, c4 = st.columns(4)
        # Sử dụng .get() để tránh lỗi nếu info bị thiếu do chặn một phần
        price = info.get('currentPrice') or hist['Close'].iloc[-1]
        c1.metric("Giá", f"{price:,.0f}")
        c2.metric("Vốn hóa", f"{info.get('marketCap', 0)/1e12:.2f}T")
        c3.metric("RSI", f"{df_plot['RSI'].iloc[-1]:.1f}")
        c4.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.1f}%")

        # --- BIỂU ĐỒ & CÁC TAB (Giữ nguyên như code cũ của bạn) ---
        tabs = st.tabs(["📈 Biểu đồ", "🧠 AI Analysis", "📊 Tài chính", "📰 Tin tức"])
        
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], name="MA20", line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], name="MA50", line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
            fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            if st.button("🚀 Phân tích AI (Gemini 2.5 Flash-Lite)"):
                # Hàm AI (giữ nguyên từ bản cũ)
                try:
                    from google import genai
                    client = genai.Client(api_key=GEMINI_API_KEY)
                    context = f"Phân tích mã {user_input}. Giá {price}. RSI {df_plot['RSI'].iloc[-1]:.2f}"
                    response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=[context])
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Lỗi AI: {e}")

        with tabs[3]:
            # Xử lý lỗi lấy tin tức nếu bị rate limit giữa chừng
            try:
                news = data['stock'].news
                if news:
                    for n in news[:5]:
                        st.write(f"**{n.get('title')}**")
                        st.caption(f"Nguồn: {n.get('publisher')}")
                        st.divider()
            except:
                st.info("Tạm thời không thể tải tin tức do giới hạn truy cập.")
    else:
        st.error("❌ Không tìm thấy dữ liệu. Vui lòng kiểm tra lại mã cổ phiếu.")
