import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import warnings
import io
from pandas.api.types import is_numeric_dtype

# --- 1. IMPORT GENAI ---
try:
    from google import genai
except ImportError:
    st.error("Thiếu thư viện google-genai. Vui lòng thêm vào requirements.txt")

# --- 2. CẤU HÌNH HỆ THỐNG ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Financial Analyzer (yfinance)", layout="wide")

# Lấy API Key từ Streamlit Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

REPORT_TYPES = {
    'financials': 'Kết quả Kinh doanh',
    'balance_sheet': 'Bảng Cân đối Kế toán',
    'cashflow': 'Lưu chuyển Tiền tệ'
}

# --- 3. HÀM XỬ LÝ DỮ LIỆU ---
def format_ticker(symbol):
    """Chuẩn hóa mã chứng khoán cho yfinance"""
    symbol = symbol.strip().upper()
    if len(symbol) == 3 and not symbol.endswith(".VN"):
        return f"{symbol}.VN"
    return symbol

@st.cache_data(show_spinner=False)
def get_financial_data(symbol, period='year'):
    """Tải dữ liệu từ yfinance với kiểm tra lỗi chặt chẽ"""
    ticker_symbol = format_ticker(symbol)
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Kiểm tra xem mã có tồn tại/có giá không
        hist = ticker.history(period="1d")
        if hist.empty:
            return None, f"Mã {ticker_symbol} không tồn tại trên Yahoo Finance."

        if period == 'year':
            data = {
                'financials': ticker.financials,
                'balance_sheet': ticker.balance_sheet,
                'cashflow': ticker.cashflow,
                'info': ticker.info
            }
        else:
            data = {
                'financials': ticker.quarterly_financials,
                'balance_sheet': ticker.quarterly_balance_sheet,
                'cashflow': ticker.quarterly_cashflow,
                'info': ticker.info
            }
        
        # Kiểm tra nếu bảng dữ liệu rỗng
        if data['financials'] is None or data['financials'].empty:
            return None, "Yahoo Finance không cung cấp báo cáo tài chính cho mã này."
            
        return data, None
    except Exception as e:
        return None, f"Lỗi kết nối: {str(e)}"

def calculate_ratios(data):
    """Tính toán các chỉ số tài chính từ dữ liệu yfinance"""
    try:
        is_df = data['financials']
        bs_df = data['balance_sheet']
        # yfinance: Index là tên chỉ tiêu, Column là ngày tháng
        ratios = pd.DataFrame()
        
        # ROE
        if 'Net Income' in is_df.index and 'Stockholders Equity' in bs_df.index:
            ratios['ROE (%)'] = (is_df.loc['Net Income'] / bs_df.loc['Stockholders Equity'] * 100).round(2)
        
        # Biên lợi nhuận gộp
        if 'Gross Profit' in is_df.index and 'Total Revenue' in is_df.index:
            ratios['Biên Lợi Nhuận Gộp (%)'] = (is_df.loc['Gross Profit'] / is_df.loc['Total Revenue'] * 100).round(2)
            
        return ratios.transpose()
    except:
        return pd.DataFrame()

# --- 4. HÀM PHÂN TÍCH AI ---
def run_ai_analysis(symbol, data):
    if not GEMINI_API_KEY:
        return "⚠️ Vui lòng thêm GEMINI_API_KEY vào Secrets để sử dụng."
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        # Gửi 3 kỳ gần nhất để AI phân tích
        context = f"Dữ liệu tài chính mã {symbol}:\n"
        context += data['financials'].iloc[:, :3].to_string()
        
        prompt = "\n\nBạn là chuyên gia tài chính. Hãy nhận xét ngắn gọn về doanh thu và lợi nhuận của công ty này qua 3 kỳ gần nhất. Đưa ra cảnh báo nếu có dấu hiệu xấu."
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[context + prompt]
        )
        return response.text
    except Exception as e:
        return f"Lỗi AI: {e}"

# --- 5. GIAO DIỆN APP ---
st.sidebar.title("🔍 Tùy chọn")
symbol = st.sidebar.text_input("Nhập mã (VNM, FPT, AAPL...):", value="VNM")
period_choice = st.sidebar.radio("Kỳ báo cáo:", ["Năm", "Quý"])
period_code = 'year' if period_choice == "Năm" else "quarter"

st.title("📊 Phân Tích Tài Chính Đa Năng")

if symbol:
    data, error_msg = get_financial_data(symbol, period_code)
    
    if error_msg:
        st.error(error_msg)
        st.info("Mẹo: Thử thêm hậu tố thủ công (ví dụ: VNM.VN) hoặc kiểm tra mã trên finance.yahoo.com")
    else:
        # Layout chính
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"📈 Tăng trưởng Doanh thu & Lợi nhuận: {symbol}")
            df_plot = data['financials'].transpose().sort_index()
            fig = go.Figure()
            if 'Total Revenue' in df_plot.columns:
                fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Total Revenue'], name="Doanh thu"))
            if 'Net Income' in df_plot.columns:
                fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Net Income'], name="Lợi nhuận ròng", line=dict(color='orange', width=3)))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🤖 AI Nhận định")
            if st.button("Chạy AI Analysis"):
                with st.spinner("AI đang xử lý..."):
                    res = run_ai_analysis(symbol, data)
                    st.write(res)
            
            st.subheader("🔢 Chỉ số cơ bản")
            ratios = calculate_ratios(data)
            if not ratios.empty:
                st.dataframe(ratios)

        # Tabs chi tiết
        st.divider()
        tabs = st.tabs([REPORT_TYPES[k] for k in REPORT_TYPES.keys()])
        for i, key in enumerate(REPORT_TYPES.keys()):
            with tabs[i]:
                st.dataframe(data[key], use_container_width=True)

        # Tải về Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for key in REPORT_TYPES.keys():
                data[key].to_excel(writer, sheet_name=key[:30])
        st.download_button(label="📥 Tải Báo cáo (Excel)", data=output.getvalue(), file_name=f"{symbol}_financials.xlsx")
