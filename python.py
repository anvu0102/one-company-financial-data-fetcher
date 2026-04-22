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
st.set_page_config(page_title="AI Financial Terminal 2026", layout="wide", page_icon="📈")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# --- 2. HÀM TẢI DỮ LIỆU CẢI TIẾN ---
@st.cache_resource(show_spinner=False)
def get_full_stock_data(symbol):
    if not symbol: return None
    
    ticker_symbol = symbol.strip().upper()
    # Danh sách mã quốc tế phổ biến để tránh thêm đuôi .VN nhầm lẫn
    global_symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'BTC-USD', 'ETH-USD', 'GOLD']
    
    # Logic xử lý mã: Nếu 3 chữ cái và không nằm trong list quốc tế -> Thêm .VN
    if len(ticker_symbol) == 3 and ticker_symbol not in global_symbols:
        ticker_symbol = f"{ticker_symbol}.VN"
    
    try:
        stock = yf.Ticker(ticker_symbol)
        # Thử lấy lịch sử giá (History là phép thử tốt nhất để biết mã có tồn tại không)
        hist = stock.history(period="2y") 
        if hist.empty:
            # Nếu vẫn trống, thử lại một lần nữa với mã gốc không thêm đuôi
            stock = yf.Ticker(symbol.strip().upper())
            hist = stock.history(period="2y")
            if hist.empty: return None
            
        return {
            "stock": stock, "hist": hist, "info": stock.info,
            "financials_y": stock.financials, "balance_sheet_y": stock.balance_sheet, "cashflow_y": stock.cashflow,
            "financials_q": stock.quarterly_financials, "balance_sheet_q": stock.quarterly_balance_sheet, "cashflow_q": stock.quarterly_cashflow
        }
    except Exception as e:
        st.sidebar.error(f"Lỗi kết nối: {str(e)}")
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
    # Hồi quy tuyến tính (Dự báo xu hướng)
    df['Idx'] = np.arange(len(df))
    last_60 = df.dropna(subset=['Close']).tail(60)
    if len(last_60) > 10:
        model = LinearRegression().fit(last_60[['Idx']], last_60['Close'])
        df['Trend'] = model.predict(df[['Idx']])
    return df

# --- 3. TRÍ TUỆ NHÂN TẠO (GEMINI-2.5-FLASH-LITE) ---
def analyze_with_ai(symbol, data_dict, info, tech_data):
    if not GEMINI_API_KEY: return "⚠️ Vui lòng cấu hình API Key trong Streamlit Secrets."
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        last_price = info.get('currentPrice', tech_data['Close'].iloc[-1])
        context = (
            f"Mã: {symbol} ({info.get('longName', 'N/A')})\n"
            f"Giá: {last_price:,.0f}. RSI: {tech_data['RSI'].iloc[-1]:.2f}\n"
            f"Tài chính (Doanh thu/LN):\n{data_dict['financials'].iloc[:, :2].to_string()}"
        )
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=[f"Phân tích kỹ thuật & cơ bản cổ phiếu này, đưa ra khuyến nghị Buy/Sell/Hold: {context}"]
        )
        return response.text
    except Exception as e: return f"❌ Lỗi AI: {str(e)}"

# --- 4. GIAO DIỆN ---
st.sidebar.title("🛠 Điều khiển")
user_input = st.sidebar.text_input("Nhập mã (VNM, FPT, AAPL, BTC-USD):", value="AAPL")
period_choice = st.sidebar.radio("Kỳ báo cáo:", ['Năm', 'Quý'])
suffix = '_y' if period_choice == 'Năm' else '_q'

if user_input:
    data = get_full_stock_data(user_input)
    
    if data:
        info, hist = data['info'], data['hist']
        df_plot = calculate_indicators(hist)
        
        # --- TOP BAR ---
        st.subheader(f"📊 {info.get('longName', user_input.upper())}")
        c1, c2, c3, c4 = st.columns(4)
        price = info.get('currentPrice', hist['Close'].iloc[-1])
        c1.metric("Giá", f"{price:,.0f}", f"{info.get('trailingPE', 0):.2f} PE")
        c2.metric("Vốn hóa", f"{info.get('marketCap', 0)/1e12:.2f}T")
        c3.metric("RSI (14)", f"{df_plot['RSI'].iloc[-1]:.1f}")
        c4.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.1f}%")

        tabs = st.tabs(["📈 Kỹ thuật & Dự báo", "🧠 Phân tích AI", "📊 Sức khỏe tài chính", "📰 Tin tức"])

        # Tab 1: Kỹ thuật
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            # Nến & MA
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], name="MA20", line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], name="MA50", line=dict(color='blue')), row=1, col=1)
            # Đường xu hướng
            if 'Trend' in df_plot:
                fig.add_trace(go.Scatter(x=df_plot.index[-60:], y=df_plot['Trend'].iloc[-60:], name="Dự báo Xu hướng", line=dict(color='red', dash='dot')), row=1, col=1)
            # RSI
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        # Tab 2: AI
        with tabs[1]:
            if st.button("🚀 Khởi chạy Gemini 2.5 Flash-Lite"):
                with st.spinner("AI đang tổng hợp dữ liệu..."):
                    res = analyze_with_ai(user_input, {"financials": data[f"financials{suffix}"]}, info, df_plot)
                    st.info(res)

        # Tab 3: Tài chính & Sức khỏe
        with tabs[2]:
            col_l, col_r = st.columns(2)
            with col_l:
                st.write("**Bảng cân đối kế toán:**")
                st.dataframe(data[f"balance_sheet{suffix}"], use_container_width=True)
            with col_r:
                st.write("**Chỉ số hiệu quả:**")
                try:
                    margin = (info.get('netIncomeToCommon', 0) / info.get('totalRevenue', 1)) * 100
                    st.write(f"- Biên lợi nhuận ròng: {margin:.2f}%")
                    st.write(f"- Tỷ lệ Nợ/Vốn CSH: {info.get('debtToEquity', 0):.2f}")
                    # Biểu đồ dòng tiền nhanh
                    cf = data[f"cashflow{suffix}"]
                    st.bar_chart(cf.iloc[:3, 0])
                except: st.write("Đang cập nhật chỉ số...")

        # Tab 4: Tin tức (Fix lỗi triệt để)
        with tabs[3]:
            try:
                news_list = data['stock'].news
                if news_list:
                    for n in news_list[:10]:
                        title = n.get('title') or n.get('content', {}).get('title', 'No Title')
                        link = n.get('link') or n.get('content', {}).get('clickThroughUrl', {}).get('url', '#')
                        if title != 'No Title':
                            st.markdown(f"**[{title}]({link})**")
                            st.caption(f"Nguồn: {n.get('publisher', 'Yahoo Finance')}")
                            st.divider()
                else: st.info("Không có tin tức mới.")
            except: st.write("Không thể lấy tin tức.")

    else:
        st.error(f"❌ Không tìm thấy dữ liệu cho '{user_input}'.")
        st.info("Mẹo: Thử thêm hậu tố (Ví dụ: 'FPT' -> 'FPT.VN', 'BTC' -> 'BTC-USD'). Kiểm tra kết nối Internet.")
