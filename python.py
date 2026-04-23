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

REPORT_TYPES = {
    'financials': 'Báo cáo Kết quả Kinh doanh',
    'balance_sheet': 'Bảng Cân đối Kế toán',
    'cashflow': 'Báo cáo Lưu chuyển Tiền tệ'
}

# --- 2. HÀM TẢI DỮ LIỆU (ĐÃ FIX LỖI) ---
@st.cache_resource(show_spinner=False)
def get_full_stock_data(symbol):
    ticker_symbol = symbol.strip().upper()
    if not ticker_symbol:
        return None
    
    try:
        stock = yf.Ticker(ticker_symbol)
        
        # 1. Kiểm tra lịch sử giá trước (Đây là dữ liệu quan trọng nhất)
        hist = stock.history(period="1y")
        if hist.empty:
            return None
        
        # 2. Lấy info một cách an toàn (yfinance thường lỗi ở đây)
        try:
            info = stock.info
        except:
            info = {}

        # 3. Trả về object dữ liệu
        return {
            "stock": stock,
            "hist": hist,
            "info": info,
            "financials_y": stock.financials,
            "balance_sheet_y": stock.balance_sheet,
            "cashflow_y": stock.cashflow,
            "financials_q": stock.quarterly_financials,
            "balance_sheet_q": stock.quarterly_balance_sheet,
            "cashflow_q": stock.quarterly_cashflow
        }
    except Exception as e:
        print(f"Lỗi hệ thống: {e}")
        return None

def calculate_indicators(df):
    df = df.copy()
    if len(df) < 14: return df
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MA
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    return df

# --- 3. AI ANALYSIS ---
def analyze_with_ai(symbol, data_dict, info, tech_data):
    if not GEMINI_API_KEY: return "⚠️ Thiếu API Key."
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Lấy giá an toàn
        last_price = info.get('currentPrice') or tech_data['Close'].iloc[-1]
        
        context = (
            f"Mã: {symbol}. Giá: {last_price}. RSI: {tech_data['RSI'].iloc[-1]:.2f}\n"
            f"Tài chính:\n{data_dict['financials'].iloc[:, :2].to_string() if not data_dict['financials'].empty else 'N/A'}"
        )
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=[f"Phân tích cổ phiếu: {context}"]
        )
        return response.text
    except Exception as e: return f"❌ Lỗi AI: {str(e)}"

# --- 4. GIAO DIỆN ---
st.title("🚀 AI Stock Analytics Pro")

# Thêm hướng dẫn nhập liệu ở Sidebar
with st.sidebar:
    st.info("💡 **Lưu ý nhập mã:**\n- Mỹ: AAPL, TSLA, BTC-USD\n- VN: HPG.VN, VNM.VN, FPT.VN")
    symbol = st.text_input("Nhập mã cổ phiếu:", value="AAPL").upper()
    period_choice = st.sidebar.radio("Kỳ báo cáo:", ['Năm', 'Quý'])

suffix = '_y' if period_choice == 'Năm' else '_q'

if symbol:
    data = get_full_stock_data(symbol)
    
    if data:
        info = data['info']
        hist = data['hist']
        df_plot = calculate_indicators(hist)
        
        # Metrics - Sử dụng .get() để tránh lỗi crash khi thiếu field
        c1, c2, c3, c4 = st.columns(4)
        curr_p = info.get('currentPrice') or hist['Close'].iloc[-1]
        currency = info.get('currency', 'Unit')
        
        c1.metric("Giá", f"{curr_p:,.0f} {currency}", f"{info.get('trailingPE', 0):.2f} P/E")
        c2.metric("Vốn hóa", f"{info.get('marketCap', 0)/1e12:.2f}T")
        c3.metric("RSI (14)", f"{df_plot['RSI'].iloc[-1]:.1f}" if 'RSI' in df_plot else "N/A")
        c4.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.1f}%")

        tabs = st.tabs(["📊 Biểu đồ", "🧠 Trợ lý AI", "📁 Tài chính", "📰 Tin tức"])

        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Giá"), row=1, col=1)
            if 'MA20' in df_plot:
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], name="MA20", line=dict(color='orange')), row=1, col=1)
            if 'RSI' in df_plot:
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
            fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            if st.button("🚀 Thực hiện nhận định"):
                current_fin = { 'financials': data[f'financials{suffix}'] }
                with st.spinner("AI đang đọc dữ liệu..."):
                    res = analyze_with_ai(symbol, current_fin, info, df_plot)
                    st.markdown(res)

        with tabs[2]:
            for key, name in REPORT_TYPES.items():
                with st.expander(name):
                    df_fin = data[f"{key}{suffix}"]
                    if df_fin is not None and not df_fin.empty:
                        st.dataframe(df_fin, use_container_width=True)
                    else:
                        st.warning("Dữ liệu này không khả dụng cho mã này.")

        with tabs[3]:
            try:
                news = data['stock'].news
                if news:
                    for item in news[:5]:
                        st.markdown(f"**[{item['title']}]({item['link']})**")
                        st.caption(f"Nguồn: {item.get('publisher', 'Yahoo Finance')}")
                        st.divider()
                else:
                    st.info("Không có tin tức mới.")
            except:
                st.error("Không thể tải tin tức.")
    else:
        st.error(f"❌ Không tìm thấy dữ liệu cho mã '{symbol}'. Nếu là chứng khoán Việt Nam, hãy thêm '.VN' (VD: HPG.VN). Nếu mã đúng, có thể Yahoo Finance đang chặn yêu cầu.")
