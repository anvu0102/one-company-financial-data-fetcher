import streamlit as st
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from pandas.api.types import is_numeric_dtype
import zipfile
import io 
import traceback

# --- 1. IMPORT THƯ VIỆN BỔ SUNG ---
try:
    from google import genai
    from google.genai.errors import APIError
except ImportError:
    st.error("Lỗi: Thư viện 'google-genai' chưa được cài đặt.")
    st.stop()
    
try:
    from vnstock import Vnstock
except ImportError:
    st.error("Lỗi: Thư viện 'vnstock3' chưa được cài đặt.")
    st.stop()

warnings.filterwarnings('ignore')

# --- CẤU HÌNH BAN ĐẦU ---
st.set_page_config(
    page_title="Debug - Phân Tích Tài Chính VN",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- KHAI BÁO BIẾN ---
DEFAULT_STOCKS = ["VNM", "FPT", "HPG", "SSI", "VIC"]
DEFAULT_STOCK_LIST = ['VNM', 'MCH', 'MSN', 'SAB', 'HAG', 'SBT', 'QNS', 'KDC', 'VHC', 'VSF']
REPORT_TYPES = {
    'balance_sheet': 'Bảng Cân đối Kế toán',
    'income_statement': 'Báo cáo Kết quả Kinh doanh',
    'cash_flow': 'Báo cáo Lưu chuyển Tiền tệ'
}
PERIOD_OPTIONS = {'year': 'Theo Năm', 'quarter': 'Theo Quý'}
SOURCE_DEFAULT = 'VCI' 

# --- SIDEBAR DEBUG OPTION ---
st.sidebar.header("🛠️ Cấu hình hệ thống")
debug_mode = st.sidebar.checkbox("Bật chế độ Debug", value=False)
if debug_mode:
    st.sidebar.warning("Chế độ Debug đang bật")

# --- HÀM TẢI DỮ LIỆU TÀI CHÍNH ---
@st.cache_data(show_spinner="Đang trích xuất dữ liệu...")
def get_financial_data(symbol, period='year', source=SOURCE_DEFAULT, is_debug=False):
    if is_debug:
        st.write(f"🔍 DEBUG: Đang gọi API cho {symbol} | Nguồn: {source} | Kỳ: {period}")
    
    financial_data = {}
    try:
        stock = Vnstock().stock(symbol=symbol, source=source)
        
        for key in REPORT_TYPES.keys():
            df = getattr(stock.finance, key)(period=period)
            financial_data[key] = df
            if is_debug and df is not None:
                st.write(f"✅ DEBUG: Đã tải {key}. Hình dạng: {df.shape}")

        if financial_data['balance_sheet'] is None or financial_data['balance_sheet'].empty:
            raise ValueError("Dữ liệu trả về rỗng từ nguồn chính")

        return financial_data
        
    except Exception as e:
        if is_debug:
            st.error(f"❌ DEBUG: Lỗi nguồn chính: {str(e)}")
            st.code(traceback.format_exc())
            
        st.warning(f"Đang thử nguồn dự phòng 'VCI' cho {symbol}...")
        try:
            stock = Vnstock().stock(symbol=symbol, source='VCI')
            return {key: getattr(stock.finance, key)(period=period) for key in REPORT_TYPES.keys()}
        except Exception as e_inner:
            st.error(f"Lỗi hoàn toàn: {e_inner}")
            return None

# --- CÁC HÀM XỬ LÝ (GIỮ NGUYÊN LOGIC) ---
def calculate_descriptive_stats(df, report_name):
    stats_list = []
    df_temp = df.reset_index() if (df is not None and not df.empty and df.index.names[0]) else df.copy()
    if df_temp is None or df_temp.empty: return pd.DataFrame()
    
    numeric_cols = [col for col in df_temp.columns if is_numeric_dtype(df_temp[col])]
    for col in numeric_cols:
        series = df_temp[col].dropna()
        if series.empty: continue
        mean_val, std_val = series.mean(), series.std()
        cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
        stats_list.append({
            'Chỉ tiêu': col, 'Trung bình': f"{mean_val:,.0f}",
            'Độ lệch chuẩn': f"{std_val:,.0f}", 'Min': f"{series.min():,.0f}",
            'Max': f"{series.max():,.0f}", 'CV': f"{cv:.2f}%"
        })
    return pd.DataFrame(stats_list)

def get_ai_analysis(stats_dfs, symbol, period, api_key, is_debug=False):
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"Phân tích tài chính {symbol} ({period}). KQKD: {stats_dfs.get('income_statement').to_markdown()}"
        if is_debug:
            with st.expander("🔍 DEBUG: Prompt gửi AI"):
                st.text(prompt)
        
        response = client.models.generate_content(model='gemini-2.5-flash-lite', contents=prompt)
        return response.text
    except Exception as e:
        if is_debug: st.code(traceback.format_exc())
        return f"Lỗi AI: {e}"

# --- GIAO DIỆN CHÍNH ---
st.title("📈 Phân Tích Báo Cáo Tài Chính VN")

analysis_mode = st.sidebar.radio("Chế độ:", ['1 Cổ phiếu', 'Danh sách'])
period = st.sidebar.radio("Kỳ:", list(PERIOD_OPTIONS.keys()), format_func=lambda x: PERIOD_OPTIONS[x])
api_key = st.sidebar.text_input("Gemini API Key", type="password")

if analysis_mode == '1 Cổ phiếu':
    symbol = st.sidebar.text_input("Mã cổ phiếu", value="VNM").upper()
    if symbol:
        data = get_financial_data(symbol, period, SOURCE_DEFAULT, debug_mode)
        
        if data:
            if debug_mode:
                with st.expander("🛠️ DEBUG: Raw DataFrames"):
                    for k, v in data.items():
                        st.write(f"Cấu trúc cột của {k}:", v.columns.tolist())
                        st.write(v.head(2))

            tabs = st.tabs(["📊 Dữ liệu", "🔢 Thống kê", "📈 Trực quan", "🤖 AI Analysis"])
            stats_dfs = {k: calculate_descriptive_stats(v, k) for k, v in data.items()}
            
            with tabs[0]:
                for k, v in data.items():
                    st.subheader(REPORT_TYPES[k])
                    st.dataframe(v, use_container_width=True)
            
            with tabs[1]:
                for k, df_s in stats_dfs.items():
                    st.subheader(f"Thống kê {REPORT_TYPES[k]}")
                    st.table(df_s)
            
            with tabs[2]:
                if not data['income_statement'].empty:
                    df_plot = data['income_statement'].iloc[::-1]
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(data=df_plot, x=df_plot.index, y='NetProfit', ax=ax, palette="viridis")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

            with tabs[3]:
                if api_key:
                    if st.button("🌟 Chạy AI Analysis"):
                        st.write(get_ai_analysis(stats_dfs, symbol, PERIOD_OPTIONS[period], api_key, debug_mode))
                else:
                    st.info("Nhập API Key ở Sidebar để dùng tính năng AI.")

else: # Chế độ danh sách
    input_list = st.sidebar.text_area("Danh sách mã", value=", ".join(DEFAULT_STOCK_LIST))
    stock_list = [s.strip().upper() for s in input_list.replace('\n', ',').split(',') if s.strip()]
    
    if st.button(f"🔍 Tải dữ liệu cho {len(stock_list)} mã"):
        all_data = {}
        prog = st.progress(0)
        for i, s in enumerate(stock_list):
            d = get_financial_data(s, period, SOURCE_DEFAULT, debug_mode)
            if d: all_data[s] = d
            prog.progress((i + 1) / len(stock_list))
            
        if all_data:
            st.success(f"Đã tải thành công {len(all_data)} mã.")
            # (Các hàm create_zip_file giữ nguyên như code cũ của bạn)
