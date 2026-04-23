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

# --- 2. HÀM HỖ TRỢ DỮ LIỆU ---

@st.cache_resource(show_spinner=False)
def get_online_data(symbol):
    ticker_symbol = symbol.strip().upper()
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
        if hist.empty: return None
        return {
            "stock": stock, "hist": hist, "info": stock.info,
            "financials_y": stock.financials, "balance_sheet_y": stock.balance_sheet, "cashflow_y": stock.cashflow,
            "financials_q": stock.quarterly_financials, "balance_sheet_q": stock.quarterly_balance_sheet, "cashflow_q": stock.quarterly_cashflow
        }
    except: return None

def process_offline_file(uploaded_file):
    """Đọc dữ liệu từ file Excel do người dùng cung cấp"""
    try:
        # Giả định file có sheet 'Price' cho dữ liệu kỹ thuật
        df_hist = pd.read_excel(uploaded_file, sheet_name='Price', index_col=0)
        df_hist.index = pd.to_datetime(df_hist.index)
        
        # Các sheet tài chính (nếu có)
        data = {
            "hist": df_hist,
            "info": {"currentPrice": df_hist['Close'].iloc[-1], "symbol": "OFFLINE"},
            "financials_y": pd.read_excel(uploaded_file, sheet_name='Financials') if 'Financials' in pd.ExcelFile(uploaded_file).sheet_names else pd.DataFrame(),
            "balance_sheet_y": pd.read_excel(uploaded_file, sheet_name='BalanceSheet') if 'BalanceSheet' in pd.ExcelFile(uploaded_file).sheet_names else pd.DataFrame(),
            "cashflow_y": pd.read_excel(uploaded_file, sheet_name='CashFlow') if 'CashFlow' in pd.ExcelFile(uploaded_file).sheet_names else pd.DataFrame(),
        }
        # Copy dữ liệu năm sang quý để tránh lỗi index nếu file offline đơn giản
        data["financials_q"] = data["financials_y"]
        data["balance_sheet_q"] = data["balance_sheet_y"]
        data["cashflow_q"] = data["cashflow_y"]
        return data
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}. File cần có ít nhất sheet 'Price' với các cột Date, Open, High, Low, Close.")
        return None

def calculate_indicators(df):
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    return df

# --- 3. AI ANALYSIS ---
def analyze_with_ai(symbol, data_dict, info, tech_data):
    if not GEMINI_API_KEY: return "⚠️ Thiếu API Key."
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        last_price = info.get('currentPrice', tech_data['Close'].iloc[-1])
        context = (
            f"Mã: {symbol}. Giá: {last_price:,.0f}. RSI: {tech_data['RSI'].iloc[-1]:.2f}\n"
            f"MA20: {tech_data['MA20'].iloc[-1]:.0f}, MA50: {tech_data['MA50'].iloc[-1]:.0f}\n"
            f"Tài chính:\n{data_dict['financials'].iloc[:, :2].to_string() if not data_dict['financials'].empty else 'N/A'}"
        )
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=[f"Phân tích chuyên sâu cổ phiếu sau: {context}"]
        )
        return response.text
    except Exception as e: return f"❌ Lỗi AI: {str(e)}"

# --- 4. GIAO DIỆN ---
st.title("🚀 AI Stock Analytics Pro")

# Sidebar
source_mode = st.sidebar.radio("Nguồn dữ liệu:", ["Trực tuyến (yfinance)", "Ngoại tuyến (Excel)"])

data = None
symbol = "OFFLINE"

if source_mode == "Trực tuyến (yfinance)":
    symbol = st.sidebar.text_input("Nhập mã ticker:", value="AAPL").upper()
    if symbol:
        data = get_online_data(symbol)
else:
    uploaded_file = st.sidebar.file_uploader("Tải file Excel (.xlsx):", type="xlsx")
    if uploaded_file:
        data = process_offline_file(uploaded_file)
        symbol = "FILE_OFFLINE"

period_choice = st.sidebar.radio("Kỳ báo cáo:", ['Năm', 'Quý'])
suffix = '_y' if period_choice == 'Năm' else '_q'

if data:
    info = data.get('info', {})
    hist = data['hist']
    df_plot = calculate_indicators(hist)
    
    # Hiển thị Metrics
    c1, c2, c3, c4 = st.columns(4)
    curr_p = info.get('currentPrice', hist['Close'].iloc[-1])
    c1.metric("Giá hiện tại", f"{curr_p:,.2f}")
    c2.metric("Vốn hóa", f"{info.get('marketCap', 0)/1e9:.2f}B")
    c3.metric("RSI (14)", f"{df_plot['RSI'].iloc[-1]:.1f}")
    c4.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.1f}%")

    tabs = st.tabs(["📊 Biểu đồ", "🧠 AI Phân tích", "📁 Tài chính", "📰 Tin tức"])

    with tabs[0]:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], 
                                   low=df_plot['Low'], close=df_plot['Close'], name="Giá"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], name="MA20", line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], name="MA50", line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
        fig.update_layout(height=600, template="plotly_white", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        if st.button("🚀 Thực hiện nhận định AI"):
            fin_key = f'financials{suffix}'
            current_fin = { 'financials': data.get(fin_key, pd.DataFrame()) }
            with st.spinner("AI đang đọc dữ liệu..."):
                res = analyze_with_ai(symbol, current_fin, info, df_plot)
                st.markdown(res)

    with tabs[2]:
        for key, name in REPORT_TYPES.items():
            df_table = data.get(f"{key}{suffix}", pd.DataFrame())
            with st.expander(name):
                st.dataframe(df_table, use_container_width=True)

    with tabs[3]:
        if source_mode == "Trực tuyến (yfinance)":
            try:
                news = data['stock'].news
                for item in news[:5]:
                    st.markdown(f"**[{item['title']}]({item['link']})**")
                    st.caption(f"Nguồn: {item.get('publisher')}")
                    st.divider()
            except:
                st.write("Không có tin tức trực tuyến.")
        else:
            st.info("Chế độ Offline không hỗ trợ tin tức thời gian thực.")
else:
    if source_mode == "Trực tuyến (yfinance)":
        st.info("Nhập mã ticker để bắt đầu.")
    else:
        st.info("Vui lòng tải file Excel có cấu trúc các sheet: 'Price', 'Financials', 'BalanceSheet', 'CashFlow'.")
