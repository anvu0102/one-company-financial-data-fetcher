import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io

# --- 1. CẤU HÌNH & THIẾT LẬP ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Financial Intelligence Pro", layout="wide")

# Lấy API Key từ Streamlit Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

REPORT_TYPES = {
    'financials': 'Báo cáo Kết quả Kinh doanh',
    'balance_sheet': 'Bảng Cân đối Kế toán',
    'cashflow': 'Báo cáo Lưu chuyển Tiền tệ'
}

# --- 2. HÀM XỬ LÝ DỮ LIỆU (SỬA LỖI CACHE) ---
@st.cache_resource(show_spinner=False)
def get_full_stock_data(symbol):
    ticker_symbol = symbol.strip().upper()
    if len(ticker_symbol) == 3: 
        ticker_symbol = f"{ticker_symbol}.VN"
    
    try:
        stock = yf.Ticker(ticker_symbol)
        # Lấy lịch sử giá 1 năm cho kỹ thuật
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
    except:
        return None

def calculate_indicators(df):
    df = df.copy()
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

# --- 3. HÀM AI ---
def analyze_with_ai(symbol, data_dict, info, tech_data):
    if not GEMINI_API_KEY:
        return "⚠️ Vui lòng cấu hình GEMINI_API_KEY để sử dụng."
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        context = f"Phân tích mã: {symbol} - {info.get('longName', '')}\n"
        context += f"Giá hiện tại: {info.get('currentPrice', 'N/A')}. RSI: {tech_data['RSI'].iloc[-1]:.2f}\n"
        context += f"--- Dữ liệu tài chính gần nhất ---\n{data_dict['financials'].iloc[:, :2].to_string()}\n"
        
        prompt = (
            "Bạn là một chuyên gia tài chính. Hãy tổng hợp dữ liệu trên và đưa ra: "
            "1. Nhận định sức khỏe tài chính. 2. Đánh giá kỹ thuật ngắn hạn (RSI/MA). "
            "3. 2 Điểm sáng & 2 Rủi ro. 4. Khuyến nghị hành động. Trả lời tiếng Việt, s
