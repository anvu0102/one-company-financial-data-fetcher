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

# --- 1. IMPORT THƯ VIỆN ---
try:
    from google import genai
    from google.genai.errors import APIError
except ImportError:
    st.error("Lỗi: Thư viện 'google-genai' chưa được cài đặt.")
    st.stop()
    
try:
    from vnstock import Vnstock
except ImportError:
    st.error("Lỗi: Thư viện 'vnstock' chưa được cài đặt.")
    st.stop()

warnings.filterwarnings('ignore')

# --- CẤU HÌNH ---
st.set_page_config(page_title="Phân Tích Tài Chính VN", layout="wide")

# Danh sách các nguồn dữ liệu để thử nghiệm theo thứ tự ưu tiên
ALL_SOURCES = ['VCI', 'TCBS', 'SSI', 'MSN', 'KBS']
PERIOD_OPTIONS = {'year': 'Theo Năm', 'quarter': 'Theo Quý'}
REPORT_TYPES = {
    'balance_sheet': 'Bảng Cân đối Kế toán',
    'income_statement': 'Báo cáo Kết quả Kinh doanh',
    'cash_flow': 'Báo cáo Lưu chuyển Tiền tệ'
}

# Sidebar Debug
st.sidebar.header("🛠️ Hệ thống")
debug_mode = st.sidebar.checkbox("Bật chế độ Debug", value=False)

# --- HÀM TẢI DỮ LIỆU ĐA NGUỒN (CORE LOGIC) ---
@st.cache_data(show_spinner=False)
def get_financial_data(symbol, period='year', is_debug=False):
    """
    Thử tất cả các nguồn dữ liệu trong ALL_SOURCES. 
    Dừng lại khi một nguồn trả về đầy đủ 3 loại báo cáo.
    """
    final_data = {}
    
    for source in ALL_SOURCES:
        if is_debug:
            st.write(f"🔍 DEBUG: Đang thử nguồn **{source}** cho mã {symbol}...")
        
        try:
            stock = Vnstock().stock(symbol=symbol, source=source)
            print(stock)
            temp_data = {}
            
            # Tải từng loại báo cáo
            for report_key in REPORT_TYPES.keys():
                df = getattr(stock.finance, report_key)(period=period)
                
                # Kiểm tra tính hợp lệ của DataFrame
                if df is not None and not df.empty:
                    temp_data[report_key] = df
                else:
                    if is_debug:
                        st.write(f"⚠️ Nguồn {source} thiếu báo cáo {report_key}")
                    break # Ngắt vòng lặp report để chuyển sang nguồn khác nếu thiếu 1 trong 3
            
            # Nếu đã tải đủ 3 báo cáo thì trả về kết quả ngay
            if len(temp_data) == 3:
                if is_debug:
                    st.success(f"✅ Thành công lấy dữ liệu từ nguồn: {source}")
                return temp_data
                
        except Exception as e:
            if is_debug:
                st.write(f"❌ Nguồn {source} thất bại: {str(e)}")
            continue # Thử nguồn tiếp theo
            
    return None

# --- HÀM HỖ TRỢ THỐNG KÊ ---
def calculate_descriptive_stats(df):
    if df is None or df.empty: return pd.DataFrame()
    stats_list = []
    # Reset index nếu index là ngày tháng để tính toán dễ hơn
    df_temp = df.reset_index() if (df.index.names[0] is not None) else df.copy()
    
    numeric_cols = [col for col in df_temp.columns if is_numeric_dtype(df_temp[col])]
    for col in numeric_cols:
        series = df_temp[col].dropna()
        if series.empty: continue
        mean_v, std_v = series.mean(), series.std()
        cv = (std_v / mean_v) * 100 if mean_v != 0 else 0
        stats_list.append({
            'Chỉ tiêu': col,
            'Trung bình': f"{mean_v:,.0f}",
            'Độ lệch chuẩn': f"{std_v:,.0f}",
            'Biến thiên (CV)': f"{cv:.2f}%"
        })
    return pd.DataFrame(stats_list)

# --- GIAO DIỆN ---
st.title("📈 Phân Tích Báo Cáo Tài Chính Đa Nguồn")
st.markdown("Hệ thống tự động chuyển đổi giữa **VCI, TCBS, SSI, MSN, KBS** nếu nguồn mặc định lỗi.")

analysis_mode = st.sidebar.radio("Chế độ:", ['1 Cổ phiếu', 'Danh sách'])
period = st.sidebar.radio("Kỳ:", list(PERIOD_OPTIONS.keys()), format_func=lambda x: PERIOD_OPTIONS[x])

if analysis_mode == '1 Cổ phiếu':
    symbol = st.sidebar.text_input("Nhập mã (VNM, HPG...):", value="VNM").upper()
    
    if symbol:
        with st.spinner(f"Đang tìm kiếm dữ liệu cho {symbol} qua các nguồn..."):
            data = get_financial_data(symbol, period, debug_mode)
            
        if data:
            tabs = st.tabs(["📊 Báo cáo", "🔢 Thống kê mô tả"])
            
            with tabs[0]:
                for key, name in REPORT_TYPES.items():
                    st.subheader(name)
                    st.dataframe(data[key], use_container_width=True)
            
            with tabs[1]:
                for key, name in REPORT_TYPES.items():
                    st.subheader(f"Phân tích {name}")
                    st.table(calculate_descriptive_stats(data[key]))
        else:
            st.error(f"Không thể tìm thấy dữ liệu cho mã **{symbol}** ở bất kỳ nguồn nào (VCI, TCBS, SSI, MSN, KBS).")

else: # Chế độ danh sách
    input_list = st.sidebar.text_area("Danh sách mã (cách nhau bởi dấu phẩy):", value="VNM, FPT, HPG")
    stock_list = [s.strip().upper() for s in input_list.replace('\n', ',').split(',') if s.strip()]
    
    if st.button("🔍 Tải dữ liệu toàn bộ danh sách"):
        results = {}
        progress_bar = st.progress(0)
        
        for i, s in enumerate(stock_list):
            st.write(f"Đang xử lý: **{s}**")
            d = get_financial_data(s, period, debug_mode)
            if d:
                results[s] = d
            progress_bar.progress((i + 1) / len(stock_list))
            
        if results:
            st.success(f"Đã tải thành công {len(results)}/{len(stock_list)} mã cổ phiếu.")
            # Logic tải ZIP tương tự các phiên bản trước...
