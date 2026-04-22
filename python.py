import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
from datetime import datetime, timedelta

# --- 1. CẤU HÌNH ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Trading Intelligence", layout="wide")

# Lấy API Key từ Streamlit Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# --- 2. HÀM XỬ LÝ DỮ LIỆU ---
@st.cache_data(show_spinner=False)
def get_stock_data(symbol):
    ticker_symbol = symbol.strip().upper()
    if len(ticker_symbol) == 3: ticker_symbol = f"{ticker_symbol}.VN"
    try:
        stock = yf.Ticker(ticker_symbol)
        # Lấy lịch sử giá 1 năm để phân tích kỹ thuật
        hist = stock.history(period="1y")
        if hist.empty: return None
        return {"stock": stock, "hist": hist, "info": stock.info}
    except:
        return None

def calculate_technical_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1+rs))
    # MA
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    return df

# --- 3. GIAO DIỆN CHÍNH ---
st.title("🚀 AI Trading & Financial Intelligence")

# Sidebar
st.sidebar.header("Tùy chỉnh")
symbol = st.sidebar.text_input("Nhập mã cổ phiếu:", value="FPT").upper()
time_range = st.sidebar.selectbox("Khoảng thời gian giá:", ["1M", "3M", "6M", "1Y", "5Y"], index=3)

range_map = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}

if symbol:
    data_bundle = get_stock_data(symbol)
    
    if data_bundle:
        stock_obj = data_bundle["stock"]
        hist = data_bundle["hist"]
        info = data_bundle["info"]
        
        # --- PHẦN 1: TỔNG QUAN THỊ TRƯỜNG ---
        c1, c2, c3, c4 = st.columns(4)
        price_now = info.get('currentPrice', hist['Close'].iloc[-1])
        price_change = price_now - hist['Close'].iloc[-2]
        pct_change = (price_change / hist['Close'].iloc[-2]) * 100
        
        c1.metric("Giá hiện tại", f"{price_now:,.0f}", f"{pct_change:.2f}%")
        c2.metric("Vốn hóa", f"{info.get('marketCap', 0)/1e12:.2f}T VNĐ")
        c3.metric("P/E", f"{info.get('trailingPE', 0):.2f}")
        c4.metric("Khối lượng GD", f"{info.get('volume', 0):,.0f}")

        # --- PHẦN 2: ĐỒ THỊ KỸ THUẬT TƯƠNG TÁC ---
        st.subheader("📈 Phân tích kỹ thuật & Xu hướng")
        df_plot = calculate_technical_indicators(hist.copy())
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, subplot_titles=('Giá & Đường MA', 'Chỉ số RSI'),
                           row_width=[0.3, 0.7])

        # Nến Nhật
        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
                                   low=df_plot['Low'], close=df_plot['Close'], name="Nến giá"), row=1, col=1)
        # Đường MA
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], name="MA20", line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], name="MA50", line=dict(color='blue')), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- PHẦN 3: SO SÁNH VỚI NGÀNH (PEER COMPARISON) ---
        st.subheader("👥 So sánh định giá")
        # Giả lập danh sách so sánh (Trong thực tế có thể dùng API ngành)
        peers = [symbol, "VNM", "MSN", "VIC"] # Ví dụ
        peer_data = []
        for p in peers:
            p_tick = yf.Ticker(f"{p}.VN" if len(p)==3 else p)
            p_info = p_tick.info
            peer_data.append({
                "Mã": p,
                "Giá": p_info.get("currentPrice", 0),
                "P/E": p_info.get("trailingPE", 0),
                "P/B": p_info.get("priceToBook", 0),
                "ROE (%)": p_info.get("returnOnEquity", 0) * 100
            })
        st.table(pd.DataFrame(peer_data))

        # --- PHẦN 4: TIN TỨC MỚI NHẤT ---
        st.subheader("📰 Tin tức liên quan")
        news = stock_obj.news[:5]
        for item in news:
            with st.expander(item['title']):
                st.write(f"Nguồn: {item['publisher']}")
                st.write(f"Link: {item['link']}")

        # --- PHẦN 5: AI CHUYÊN SÂU ---
        st.markdown("---")
        st.subheader("🧠 Trợ lý chiến lược AI")
        if st.button("🔍 Tổng hợp chiến lược đầu tư"):
            if not GEMINI_API_KEY:
                st.warning("Vui lòng thêm GEMINI_API_KEY")
            else:
                from google import genai
                client = genai.Client(api_key=GEMINI_API_KEY)
                
                # Gom dữ liệu kỹ thuật và cơ bản cho AI
                last_rsi = df_plot['RSI'].iloc[-1]
                last_close = df_plot['Close'].iloc[-1]
                ma20 = df_plot['MA20'].iloc[-1]
                
                ai_prompt = (
                    f"Phân tích cổ phiếu {symbol}. Giá đóng cửa: {last_close}. RSI: {last_rsi:.2f}. "
                    f"Đường MA20 hiện tại là {ma20:.2f}. P/E là {info.get('trailingPE')}. "
                    "Hãy đưa ra: 1. Đánh giá xu hướng ngắn hạn. 2. Vùng hỗ trợ/kháng cự dự kiến. "
                    "3. Lời khuyên (Mua/Bán/Theo dõi) dựa trên sự kết hợp giữa PT kỹ thuật và cơ bản."
                )
                
                with st.spinner("AI đang tính toán chiến lược..."):
                    response = client.models.generate_content(model="gemini-2.0-flash", contents=[ai_prompt])
                    st.success("Khuyến nghị của AI:")
                    st.write(response.text)

    else:
        st.error("Không thể lấy dữ liệu. Kiểm tra lại mã cổ phiếu.")
