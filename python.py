import streamlit as st
import pandas as pd
import yfinance as yf
import warnings
from pandas.api.types import is_numeric_dtype
import io

# --- 1. IMPORT GENAI ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("Lỗi: Thư viện 'google-genai' chưa được cài đặt. Vui lòng thêm vào requirements.txt")
    st.stop()

# --- 2. CẤU HÌNH & THIẾT LẬP ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Phân Tích Tài Chính", layout="wide")

REPORT_TYPES = {
    'financials': 'Báo cáo Kết quả Kinh doanh',
    'balance_sheet': 'Bảng Cân đối Kế toán',
    'cashflow': 'Báo cáo Lưu chuyển Tiền tệ'
}

# Lấy API Key từ Streamlit Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# --- 3. HÀM XỬ LÝ AI ---
def analyze_with_ai(symbol, data_dict):
    """Gửi dữ liệu tài chính cho Gemini để phân tích"""
    if not GEMINI_API_KEY:
        return "⚠️ Vui lòng cấu hình GEMINI_API_KEY trong Secrets để sử dụng tính năng này."
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Chuẩn bị dữ liệu tóm tắt để gửi cho AI (tránh quá tải token)
        context = f"Phân tích báo cáo tài chính cho mã cổ phiếu: {symbol}\n\n"
        for key, name in REPORT_TYPES.items():
            df_summary = data_dict[key].iloc[:, :2] # Lấy 2 kỳ gần nhất
            context += f"--- {name} (2 kỳ gần nhất) ---\n{df_summary.to_string()}\n\n"
        
        prompt = (
            f"Bạn là một chuyên gia phân tích tài chính. Dựa trên dữ liệu sau của {symbol}, "
            "hãy tóm tắt 3 điểm sáng và 3 rủi ro chính. Đưa ra nhận xét về sức khỏe tài chính hiện tại. "
            "Trả lời bằng tiếng Việt, ngắn gọn, súc tích."
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", # Hoặc gemini-1.5-flash
            contents=[context + prompt]
        )
        return response.text
    except Exception as e:
        return f"❌ Lỗi khi gọi AI: {str(e)}"

# --- 4. HÀM TẢI DỮ LIỆU ---
@st.cache_data(show_spinner=False)
def get_financial_data_yf(symbol, period='year'):
    ticker_symbol = symbol.strip().upper()
    if len(ticker_symbol) == 3:
        ticker_symbol = f"{ticker_symbol}.VN"
    
    try:
        stock = yf.Ticker(ticker_symbol)
        temp_data = {}
        if period == 'year':
            temp_data['balance_sheet'] = stock.balance_sheet
            temp_data['financials'] = stock.financials
            temp_data['cashflow'] = stock.cashflow
        else:
            temp_data['balance_sheet'] = stock.quarterly_balance_sheet
            temp_data['financials'] = stock.quarterly_financials
            temp_data['cashflow'] = stock.quarterly_cashflow

        if temp_data['financials'] is not None and not temp_data['financials'].empty:
            return temp_data
        return None
    except:
        return None

# --- 5. GIAO DIỆN ---
st.title("🤖 AI Phân Tích Tài Chính (Yahoo Finance + Gemini)")

symbol = st.sidebar.text_input("Nhập mã cổ phiếu:", value="VNM").upper()
period_choice = st.sidebar.radio("Kỳ báo cáo:", ['Năm', 'Quý'])
period_code = 'year' if period_choice == 'Năm' else 'quarter'

if symbol:
    with st.spinner(f"Đang tải dữ liệu {symbol}..."):
        data = get_financial_data_yf(symbol, period_code)
        
    if data:
        # Phần 1: Phân tích AI
        st.subheader("🧠 Nhận định từ AI (Gemini)")
        if st.button("🚀 Chạy phân tích AI"):
            with st.spinner("AI đang đọc báo cáo..."):
                analysis = analyze_with_ai(symbol, data)
                st.info(analysis)
        
        # Phần 2: Hiển thị bảng biểu
        tabs = st.tabs(["📊 Báo cáo", "🔢 Tải về"])
        with tabs[0]:
            for key, name in REPORT_TYPES.items():
                with st.expander(name):
                    st.dataframe(data[key], use_container_width=True)
        
        with tabs[1]:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for key in REPORT_TYPES.keys():
                    data[key].to_excel(writer, sheet_name=key[:30])
            st.download_button("📥 Tải Excel", output.getvalue(), f"{symbol}_data.xlsx")
    else:
        st.error("Không tìm thấy dữ liệu.")
