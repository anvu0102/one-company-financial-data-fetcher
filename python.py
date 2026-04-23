import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io

# --- 1. CẤU HÌNH ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Financial Analyst Pro", layout="wide", page_icon="📈")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# --- 2. HÀM TẢI DỮ LIỆU ---
@st.cache_resource(show_spinner=False)
def get_full_stock_data(symbol):
    ticker_symbol = symbol.strip().upper()
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
        if hist.empty: return None
        
        # Lấy info an toàn
        try: info = stock.info
        except: info = {}

        return {
            "stock": stock, "hist": hist, "info": info,
            "financials_y": stock.financials, "balance_sheet_y": stock.balance_sheet, "cashflow_y": stock.cashflow,
            "financials_q": stock.quarterly_financials, "balance_sheet_q": stock.quarterly_balance_sheet, "cashflow_q": stock.quarterly_cashflow
        }
    except: return None

def calculate_indicators(df):
    df = df.copy()
    if len(df) < 20: return df
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    return df

# --- 3. GIAO DIỆN ---
st.title("🚀 AI Stock Analytics Pro")

with st.sidebar:
    st.markdown("### 🔍 Tra cứu")
    symbol = st.text_input("Nhập mã (VD: AAPL, HPG.VN):", value="AAPL").upper()
    period_choice = st.radio("Kỳ báo cáo:", ['Năm', 'Quý'])

suffix = '_y' if period_choice == 'Năm' else '_q'

if symbol:
    data = get_full_stock_data(symbol)
    if data:
        info, hist = data['info'], data['hist']
        df_plot = calculate_indicators(hist)
        
        tabs = st.tabs(["📊 Biểu đồ", "🧠 AI Analysis", "📁 Tài chính", "📰 Tin tức"])

        # ... (Phần Biểu đồ và Tài chính giữ nguyên như code trước) ...
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Giá"), row=1, col=1)
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        # --- FIX LỖI TIN TỨC TẠI ĐÂY ---
        with tabs[3]:
            st.subheader(f"📰 Tin tức & Sự kiện: {symbol}")
            try:
                # Lấy tin tức từ yfinance
                raw_news = data['stock'].news
                
                if raw_news and len(raw_news) > 0:
                    for item in raw_news[:8]: # Hiển thị tối đa 8 tin
                        # Xử lý các định dạng lồng nhau khác nhau của Yahoo API
                        title = item.get('title') or item.get('content', {}).get('title', 'Bản tin tài chính')
                        link = item.get('link') or item.get('content', {}).get('clickThroughUrl', {}).get('url', '#')
                        publisher = item.get('publisher') or item.get('content', {}).get('publisher', 'Yahoo Finance')
                        
                        # Hiển thị tin tức
                        with st.container():
                            col_news, col_spacer = st.columns([0.9, 0.1])
                            with col_news:
                                st.markdown(f"#### [{title}]({link})")
                                st.caption(f"📢 Nguồn: {publisher}")
                                st.divider()
                else:
                    st.info("Hiện không tìm thấy tin tức mới từ Yahoo Finance cho mã này.")
                    # Thêm link dự phòng nếu không thấy tin
                    st.markdown(f"[Xem tin tức {symbol} trên Google Finance](https://www.google.com/finance/quote/{symbol.replace('.VN', ':VNSE')})")

            except Exception as e:
                st.error(f"Không thể kết nối API tin tức: {e}")
    else:
        st.error("Không tìm thấy dữ liệu. Hãy kiểm tra lại mã Ticker.")
