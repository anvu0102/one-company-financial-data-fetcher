import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from pandas.api.types import is_numeric_dtype
import io
import traceback

# --- 1. CẤU HÌNH & THIẾT LẬP ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Phân Tích Tài Chính Quốc Tế & VN", layout="wide")

REPORT_TYPES = {
    'balance_sheet': 'Bảng Cân đối Kế toán',
    'financials': 'Báo cáo Kết quả Kinh doanh',
    'cashflow': 'Báo cáo Lưu chuyển Tiền tệ'
}

# Sidebar Debug
st.sidebar.header("🛠️ Hệ thống")
debug_mode = st.sidebar.checkbox("Bật chế độ Debug", value=False)

# --- HÀM TẢI DỮ LIỆU QUA YFINANCE ---
@st.cache_data(show_spinner=False)
def get_financial_data_yf(symbol, period='year', is_debug=False):
    """
    Sử dụng yfinance để tải báo cáo tài chính.
    Lưu ý: Mã VN cần thêm đuôi .VN
    """
    # Xử lý ticker (tự động thêm .VN nếu là mã 3 chữ cái)
    ticker_symbol = symbol.strip().upper()
    if len(ticker_symbol) == 3:
        ticker_symbol = f"{ticker_symbol}.VN"
    
    if is_debug:
        st.write(f"🔍 DEBUG: Đang truy vấn Yahoo Finance cho: **{ticker_symbol}**")

    try:
        stock = yf.Ticker(ticker_symbol)
        temp_data = {}

        # Tải dữ liệu theo kỳ (Năm hoặc Quý)
        if period == 'year':
            temp_data['balance_sheet'] = stock.balance_sheet
            temp_data['financials'] = stock.financials
            temp_data['cashflow'] = stock.cashflow
        else:
            temp_data['balance_sheet'] = stock.quarterly_balance_sheet
            temp_data['financials'] = stock.quarterly_financials
            temp_data['cashflow'] = stock.quarterly_cashflow

        # Kiểm tra xem có dữ liệu không
        valid_count = 0
        for key in REPORT_TYPES.keys():
            if temp_data[key] is not None and not temp_data[key].empty:
                valid_count += 1
            else:
                if is_debug: st.write(f"⚠️ Thiếu báo cáo: {key}")

        if valid_count > 0:
            return temp_data
        return None

    except Exception as e:
        if is_debug:
            st.error(f"❌ Lỗi yfinance: {str(e)}")
        return None

# --- HÀM HỖ TRỢ THỐNG KÊ ---
def calculate_descriptive_stats(df):
    if df is None or df.empty: return pd.DataFrame()
    
    # yfinance trả về index là tên chỉ tiêu, cột là ngày tháng
    # Ta cần chuyển vị (transpose) để tính toán theo chỉ tiêu
    df_t = df.transpose()
    
    stats_list = []
    numeric_cols = [col for col in df_t.columns if is_numeric_dtype(df_t[col])]
    
    for col in numeric_cols:
        series = df_t[col].dropna()
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
st.title("📈 Phân Tích Báo Cáo Tài Chính (Yahoo Finance)")
st.markdown("Hệ thống sử dụng dữ liệu từ **Yahoo Finance API**. Đối với mã Việt Nam, vui lòng nhập mã 3 chữ cái (ví dụ: VNM, FPT).")

analysis_mode = st.sidebar.radio("Chế độ:", ['1 Cổ phiếu', 'Danh sách'])
period_choice = st.sidebar.radio("Kỳ báo cáo:", ['Năm', 'Quý'])
period_code = 'year' if period_choice == 'Năm' else 'quarter'

if analysis_mode == '1 Cổ phiếu':
    symbol = st.sidebar.text_input("Nhập mã (VNM, FPT, AAPL...):", value="VNM").upper()
    
    if symbol:
        with st.spinner(f"Đang tải dữ liệu cho {symbol}..."):
            data = get_financial_data_yf(symbol, period_code, debug_mode)
            
        if data:
            tabs = st.tabs(["📊 Báo cáo tài chính", "🔢 Phân tích thống kê"])
            
            with tabs[0]:
                for key, name in REPORT_TYPES.items():
                    with st.expander(name, expanded=True):
                        st.dataframe(data[key], use_container_width=True)
            
            with tabs[1]:
                for key, name in REPORT_TYPES.items():
                    st.subheader(f"Thống kê: {name}")
                    stats_df = calculate_descriptive_stats(data[key])
                    if not stats_df.empty:
                        st.table(stats_df)
                    else:
                        st.info("Không đủ dữ liệu số để thống kê.")
        else:
            st.error(f"Không thể tìm thấy dữ liệu cho mã **{symbol}**. Lưu ý: Một số mã cổ phiếu VN có thể chưa được Yahoo Finance cập nhật đầy đủ báo cáo.")

else: # Chế độ danh sách
    input_list = st.sidebar.text_area("Danh sách mã (cách nhau bởi dấu phẩy):", value="VNM, FPT, HPG")
    stock_list = [s.strip().upper() for s in input_list.replace('\n', ',').split(',') if s.strip()]
    
    if st.button("🔍 Tải dữ liệu toàn bộ danh sách"):
        results = {}
        progress_bar = st.progress(0)
        
        for i, s in enumerate(stock_list):
            st.write(f"Đang xử lý: **{s}**")
            d = get_financial_data_yf(s, period_code, debug_mode)
            if d:
                results[s] = d
            progress_bar.progress((i + 1) / len(stock_list))
            
        if results:
            st.success(f"Đã tải thành công dữ liệu cho {len(results)} mã.")
            
            # Tạo file Excel để tải về
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for s, d in results.items():
                    for r_key, r_name in REPORT_TYPES.items():
                        sheet_name = f"{s}_{r_key[:10]}"
                        d[r_key].to_excel(writer, sheet_name=sheet_name)
            
            st.download_button(
                label="📥 Tải toàn bộ dữ liệu (Excel)",
                data=output.getvalue(),
                file_name="financial_data_yf.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
