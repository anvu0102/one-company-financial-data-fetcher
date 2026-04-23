import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io

# --- 1. CẤU HÌNH HỆ THỐNG ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Financial Analyst Pro", layout="wide", page_icon="🚀")

# Lấy API Key từ Streamlit Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

REPORT_TYPES = {
    'financials': 'Báo cáo Kết quả Kinh doanh',
    'balance_sheet': 'Bảng Cân đối Kế toán',
    'cashflow': 'Báo cáo Lưu chuyển Tiền tệ'
}

# --- 2. HÀM TẢI DỮ LIỆU ---
@st.cache_resource(show_spinner=False)
def get_full_stock_data(symbol):
    ticker_symbol = symbol.strip().upper()
    # Tự động thêm .VN cho mã Việt Nam (3 chữ cái), ngoại trừ các mã US phổ biến
    us_indices = ['AAPL', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'BTC-USD', 'NFLX']
    if len(ticker_symbol) == 3 and ticker_symbol not in us_indices: 
        ticker_symbol = f"{ticker_symbol}.VN"
    
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
        if hist.empty: return None
        
        return {
            "stock": stock,
            "hist": hist,
            "info": stock.info,
            "financials_y": stock.financials,
            "balance_sheet_y": stock.balance_sheet,
            "cashflow_y": stock.cashflow,
            "financials_q": stock.quarterly_financials,
            "balance_sheet_q": stock.quarterly_balance_sheet,
            "cashflow_q": stock.quarterly_cashflow
        }
    except Exception:
        return None

def calculate_indicators(df):
    df = df.copy()
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean() # THÊM MA 50
    return df

# --- 3. TRÍ TUỆ NHÂN TẠO (GEMINI-2.5-FLASH-LITE) ---
def analyze_with_ai(symbol, data_dict, info, tech_data):
    if not GEMINI_API_KEY:
        return "⚠️ Vui lòng cấu hình GEMINI_API_KEY trong Streamlit Secrets."
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        last_price = info.get('currentPrice', tech_data['Close'].iloc[-1])
        rsi_val = tech_data['RSI'].iloc[-1]
        
        context = (
            f"Phân tích mã: {symbol} ({info.get('longName', '')})\n"
            f"Giá hiện tại: {last_price:,}. RSI: {rsi_val:.2f}\n"
            f"Xu hướng: MA20={tech_data['MA20'].iloc[-1]:.2f}, MA50={tech_data['MA50'].iloc[-1]:.2f}\n"
            f"--- Tài chính ---\n{data_dict['financials'].iloc[:, :2].to_string()}"
        )
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=[f"Đóng vai chuyên gia tài chính phân tích kỹ thuật và cơ bản cho dữ liệu sau: {context}"]
        )
        return response.text
    except Exception as e:
        return f"❌ Lỗi AI: {str(e)}"

# --- 4. GIAO DIỆN NGƯỜI DÙNG ---
st.title("📈 AI Trading Intelligence Pro")

symbol = st.sidebar.text_input("Nhập mã cổ phiếu (VD: AAPL, FPT, VNM):", value="AAPL").upper()
period_choice = st.sidebar.radio("Kỳ báo cáo tài chính:", ['Năm', 'Quý'])
suffix = '_y' if period_choice == 'Năm' else '_q'

if symbol:
    data = get_full_stock_data(symbol)
    if data:
        info = data['info']
        hist = data['hist']
        df_plot = calculate_indicators(hist)
        
        # Dashboard Chỉ số nhanh
        c1, c2, c3, c4 = st.columns(4)
        curr_p = info.get('currentPrice', hist['Close'].iloc[-1])
        c1.metric("Giá", f"{curr_p:,.0f}", f"{info.get('trailingPE', 0):.2f} P/E")
        c2.metric("Vốn hóa", f"{info.get('marketCap', 0)/1e12:.2f}T")
        c3.metric("RSI (14)", f"{df_plot['RSI'].iloc[-1]:.1f}")
        c4.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.1f}%")

        tabs = st.tabs(["📊 Biểu đồ kỹ thuật", "🧠 Trợ lý AI", "📁 Báo cáo tài chính", "📰 Tin tức & So sánh"])

        # TAB 1: BIỂU ĐỒ KỸ THUẬT (MA20, MA50 & RSI THRESHOLD)
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            # Candle Stick
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], 
                                       low=df_plot['Low'], close=df_plot['Close'], name="Giá"), row=1, col=1)
            # MA 20 & MA 50
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], name="MA20 (Ngắn hạn)", line=dict(color='orange', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], name="MA50 (Trung hạn)", line=dict(color='blue', width=1.5)), row=1, col=1)
            
            # RSI & THRESHOLDS
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Quá mua (70)", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Quá bán (30)", row=2, col=1)
            
            fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

        # TAB 2: AI ANALYSIS
        with tabs[1]:
            st.subheader("🤖 Nhận định từ Gemini 2.5 Flash-Lite")
            if st.button("🚀 Thực hiện phân tích AI"):
                current_fin = { 'financials': data[f'financials{suffix}'] }
                with st.spinner("AI đang tính toán..."):
                    res = analyze_with_ai(symbol, current_fin, info, df_plot)
                    st.markdown(res)

        # TAB 3: FINANCIAL DATA
        with tabs[2]:
            for key, name in REPORT_TYPES.items():
                with st.expander(name):
                    st.dataframe(data[f"{key}{suffix}"], use_container_width=True)
            
            # Download Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for key in REPORT_TYPES.keys():
                    data[f"{key}{suffix}"].to_excel(writer, sheet_name=key[:30])
            st.download_button("📥 Tải dữ liệu Excel", output.getvalue(), f"{symbol}_financials.xlsx")

        # TAB 4: NEWS & COMPARISON (FIXED LOGIC)
        with tabs[3]:
            col_l, col_r = st.columns([1, 2])
            with col_l:
                st.subheader("👥 Định giá & So sánh")
                comp = {
                    "Chỉ số": ["P/E", "P/B", "Giá mục tiêu", "Biên LN Gộp"],
                    "Giá trị": [
                        str(info.get('trailingPE', 'N/A')), 
                        str(info.get('priceToBook', 'N/A')),
                        str(info.get('targetMeanPrice', 'N/A')),
                        f"{info.get('grossMargins', 0)*100:.2f}%"
                    ]
                }
                st.table(pd.DataFrame(comp).astype(str))
            
            with col_r:
                st.subheader("📰 Tin tức mới nhất")
                try:
                    raw_news = data['stock'].news
                    if isinstance(raw_news, list) and len(raw_news) > 0:
                        for item in raw_news[:8]:
                            if isinstance(item, dict):
                                title = item.get('title') or item.get('content', {}).get('title', 'No Title')
                                link = item.get('link') or item.get('content', {}).get('clickThroughUrl', {}).get('url', '#')
                                pub = item.get('publisher', 'Nguồn uy tín')
                                
                                if title != 'No Title':
                                    with st.container():
                                        st.markdown(f"**[{title}]({link})**")
                                        st.caption(f"Nguồn: {pub}")
                                        st.divider()
                    else:
                        st.info("💡 Không tìm thấy tin tức mới cho mã này trên Yahoo Finance.")
                except Exception as e:
                    st.error(f"Lỗi tải tin: {e}")
    else:
        st.error("⚠️ Không tìm thấy dữ liệu. Vui lòng kiểm tra lại mã cổ phiếu.")
