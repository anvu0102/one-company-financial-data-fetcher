import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import io

# --- 1. CẤU HÌNH ---
st.set_page_config(page_title="Hệ thống Phân tích Tài chính 4.0", layout="wide")

# Lấy API Key từ Secrets cho GenAI
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
try:
    from google import genai
except ImportError:
    st.error("Thiếu thư viện google-genai")

# --- 2. HÀM HỖ TRỢ DỮ LIỆU ---
def format_ticker(symbol):
    symbol = symbol.strip().upper()
    return f"{symbol}.VN" if len(symbol) == 3 else symbol

@st.cache_data
def get_data(symbol, period='year'):
    try:
        ticker = yf.Ticker(format_ticker(symbol))
        if period == 'year':
            return {
                'is': ticker.financials,
                'bs': ticker.balance_sheet,
                'cf': ticker.cashflow,
                'info': ticker.info
            }
        else:
            return {
                'is': ticker.quarterly_financials,
                'bs': ticker.quarterly_balance_sheet,
                'cf': ticker.quarterly_cashflow,
                'info': ticker.info
            }
    except:
        return None

# --- 3. TÍNH NĂNG MỚI: TÍNH TOÁN CHỈ SỐ TÀI CHÍNH ---
def calculate_ratios(data):
    """Tính toán các chỉ số cơ bản từ báo cáo yfinance"""
    is_df = data['is']
    bs_df = data['bs']
    
    ratios = pd.DataFrame()
    try:
        # Ví dụ tính ROE = Net Income / Total Stockholders Equity
        net_income = is_df.loc['Net Income']
        equity = bs_df.loc['Stockholders Equity']
        ratios['ROE (%)'] = (net_income / equity * 100).round(2)
        
        # Biên lợi nhuận gộp = Gross Profit / Total Revenue
        ratios['Biên Lợi Nhuận Gộp (%)'] = (is_df.loc['Gross Profit'] / is_df.loc['Total Revenue'] * 100).round(2)
        
        # Chỉ số thanh toán hiện hành = Current Assets / Current Liabilities
        ratios['Thanh toán hiện hành (x)'] = (bs_df.loc['Current Assets'] / bs_df.loc['Current Liabilities']).round(2)
    except:
        pass
    return ratios.transpose()

# --- 4. GIAO DIỆN CHÍNH ---
st.title("🚀 Hệ thống Phân tích Tài chính Thông minh")

menu = st.sidebar.selectbox("Tính năng:", ["Phân tích chi tiết", "So sánh mã (Peer)", "AI Chatbot"])

# --- FEATURE 1: PHÂN TÍCH CHI TIẾT & BIỂU ĐỒ ---
if menu == "Phân tích chi tiết":
    symbol = st.sidebar.text_input("Nhập mã:", "VNM")
    period = st.sidebar.radio("Kỳ:", ["Năm", "Quý"])
    
    data = get_data(symbol, "year" if period == "Năm" else "quarter")
    
    if data and not data['is'].empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Biểu đồ Doanh thu & Lợi nhuận: {symbol}")
            # Vẽ biểu đồ Plotly
            df_plot = data['is'].transpose()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Total Revenue'], name='Doanh thu'))
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Net Income'], name='Lợi nhuận ròng', line=dict(color='orange', width=3)))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Chỉ số tài chính")
            ratio_df = calculate_ratios(data)
            st.table(ratio_df)

        tabs = st.tabs(["Bảng cân đối", "Kết quả KD", "Lưu chuyển tiền"])
        tabs[0].dataframe(data['bs'])
        tabs[1].dataframe(data['is'])
        tabs[2].dataframe(data['cf'])
    else:
        st.error("Không tìm thấy dữ liệu.")

# --- FEATURE 2: SO SÁNH PEER COMPARISON ---
elif menu == "So sánh mã (Peer)":
    st.subheader("So sánh các mã cùng ngành")
    compare_list = st.text_input("Nhập các mã (cách nhau dấu phẩy):", "VNM, MSN, SAB")
    symbols = [s.strip() for s in compare_list.split(",")]
    
    comp_data = []
    for s in symbols:
        ticker = yf.Ticker(format_ticker(s))
        info = ticker.info
        comp_data.append({
            "Mã": s,
            "Giá": info.get("currentPrice"),
            "P/E": info.get("trailingPE"),
            "P/B": info.get("priceToBook"),
            "Vốn hóa (Tỷ)": info.get("marketCap", 0) / 10**9,
            "ROE (%)": info.get("returnOnEquity", 0) * 100
        })
    
    st.table(pd.DataFrame(comp_data))

# --- FEATURE 3: AI CHATBOT TƯ VẤN ---
elif menu == "AI Chatbot":
    st.subheader("🤖 Trợ lý ảo Gemini Finance")
    user_q = st.text_input("Hỏi AI về cổ phiếu (VD: VNM có tốt để đầu tư dài hạn không?):")
    
    if st.button("Hỏi AI"):
        if GEMINI_API_KEY:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[f"Dựa trên kiến thức tài chính, hãy trả lời: {user_q}"]
            )
            st.markdown(response.text)
        else:
            st.warning("Vui lòng cấu hình API Key.")
