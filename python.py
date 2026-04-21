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

# --- 1. IMPORT THƯ VIỆN BỔ SUNG ---
try:
    from google import genai
    from google.genai.errors import APIError
except ImportError:
    st.error("Lỗi: Thư viện 'google-genai' chưa được cài đặt. Vui lòng chạy `pip install google-genai`.")
    st.stop()
    
try:
    # Sử dụng vnstock3 để tránh lỗi 404 của bản cũ
    from vnstock3 import Vnstock
except ImportError:
    st.error("Lỗi: Thư viện 'vnstock3' chưa được cài đặt. Vui lòng chạy `pip install vnstock3`.")
    st.stop()

# --- SỬA LỖI CẤU HÌNH ---
warnings.filterwarnings('ignore')

# --- CẤU HÌNH BAN ĐẦU ---
st.set_page_config(
    page_title="Phân Tích Dữ Liệu Báo Cáo Tài Chính Việt Nam",
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
PERIOD_OPTIONS = {
    'year': 'Theo Năm',
    'quarter': 'Theo Quý'
}
# Đổi nguồn từ KBS sang VCI hoặc TCBS để tránh lỗi 404
SOURCE_DEFAULT = 'VCI' 

# --- HÀM TẢI DỮ LIỆU TÀI CHÍNH ---
@st.cache_data(show_spinner="Đang trích xuất dữ liệu...")
def get_financial_data(symbol, period='year', source=SOURCE_DEFAULT):
    """Tải dữ liệu bằng vnstock3 với cơ chế fallback nguồn dữ liệu."""
    st.info(f"Đang trích xuất: **{symbol}** (Nguồn: {source})...")
    financial_data = {}
    
    try:
        # Khởi tạo API với vnstock3
        stock = Vnstock().stock(symbol=symbol, source=source)
        
        financial_data['balance_sheet'] = stock.finance.balance_sheet(period=period)
        financial_data['income_statement'] = stock.finance.income_statement(period=period)
        financial_data['cash_flow'] = stock.finance.cash_flow(period=period)

        # Kiểm tra nếu dữ liệu rỗng
        if financial_data['balance_sheet'] is None or financial_data['balance_sheet'].empty:
            raise ValueError("Dữ liệu trả về rỗng")

        return financial_data
        
    except Exception as e:
        st.warning(f"Nguồn {source} lỗi cho mã {symbol}. Đang thử nguồn dự phòng 'TCBS'...")
        try:
            # Fallback sang nguồn TCBS nếu nguồn chính lỗi 404
            stock = Vnstock().stock(symbol=symbol, source='TCBS')
            financial_data['balance_sheet'] = stock.finance.balance_sheet(period=period)
            financial_data['income_statement'] = stock.finance.income_statement(period=period)
            financial_data['cash_flow'] = stock.finance.cash_flow(period=period)
            return financial_data
        except Exception as e_inner:
            st.error(f"Lỗi hoàn toàn cho mã {symbol}: {e_inner}")
            return None

@st.cache_data(show_spinner="Đang tải danh sách...")
def get_all_financial_data(stock_list, period='year', source=SOURCE_DEFAULT):
    all_data = {}
    status_text = st.empty()
    for i, symbol in enumerate(stock_list):
        status_text.info(f"Tiến độ: {i + 1}/{len(stock_list)} - {symbol}")
        data = get_financial_data(symbol, period, source)
        if data:
            all_data[symbol] = data
    status_text.success(f"Hoàn tất! Đã tải {len(all_data)} mã.")
    return all_data

# --- HÀM XUẤT FILE ---
def to_excel(df_to_save, name):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        sheet_name = name.replace(' ', '_')[:30]
        df_to_save.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()

def create_combined_excel(symbol, financial_data):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        has_data = False
        for key, name in REPORT_TYPES.items():
            df = financial_data.get(key)
            if df is not None and not df.empty:
                df_to_save = df.reset_index() if df.index.names[0] else df.copy()
                df_to_save.to_excel(writer, index=False, sheet_name=name[:30])
                has_data = True
    return output.getvalue() if has_data else None

def create_txt_content(symbol, financial_data, period):
    content = f"===== BÁO CÁO TỔNG HỢP {symbol} ({PERIOD_OPTIONS[period]}) =====\n\n"
    for key, name in REPORT_TYPES.items():
        df = financial_data.get(key)
        if df is not None and not df.empty:
            content += f"\n--- {name} ---\n{df.to_string(index=False)}\n"
    return content

def create_zip_file(all_financial_data, period, mode='excel'):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
        for symbol, data in all_financial_data.items():
            if mode == 'excel':
                content = create_combined_excel(symbol, data)
                ext = "xlsx"
            else:
                content = create_txt_content(symbol, data, period).encode('utf-8')
                ext = "txt"
            
            if content:
                zip_file.writestr(f'BC_Tai_Chinh_{symbol}.{ext}', content)
    return zip_buffer.getvalue()

def calculate_descriptive_stats(df, report_name):
    stats_list = []
    df_temp = df.reset_index() if df.index.names[0] else df.copy()
    numeric_cols = [col for col in df_temp.columns if is_numeric_dtype(df_temp[col])]
    time_col = next((c for c in ['id', 'ReportDate', 'Period'] if c in df_temp.columns), df_temp.columns[0])

    for col in numeric_cols:
        series = df_temp[col].dropna()
        if series.empty: continue
        
        mean_val, std_val = series.mean(), series.std()
        cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
        
        stats_list.append({
            'Chỉ tiêu': col,
            'Trung bình': f"{mean_val:,.0f}",
            'Độ lệch chuẩn': f"{std_val:,.0f}",
            'Min': f"{series.min():,.0f}",
            'Max': f"{series.max():,.0f}",
            'Hệ số biến thiên (CV)': f"{cv:.2f}%"
        })
    return pd.DataFrame(stats_list)

def get_ai_analysis(stats_dfs, symbol, period, api_key):
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
        Phân tích tài chính cho {symbol} ({period}).
        Báo cáo KQKD: {stats_dfs.get('income_statement').to_markdown()}
        Bảng Cân đối: {stats_dfs.get('balance_sheet').to_markdown()}
        Yêu cầu: Viết 4-6 đoạn tiếng Việt đánh giá tăng trưởng, rủi ro nợ và sức khỏe tài chính.
        """
        response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        return response.text
    except Exception as e:
        return f"Lỗi AI: {e}"

# --- GIAO DIỆN ---
st.title("📈 Phân Tích Báo Cáo Tài Chính VN (Vnstock3 Edition)")

# Sidebar
analysis_mode = st.sidebar.radio("Chế độ:", ['1 Cổ phiếu', 'Danh sách'])
period = st.sidebar.radio("Kỳ:", list(PERIOD_OPTIONS.keys()), format_func=lambda x: PERIOD_OPTIONS[x])
api_key = st.sidebar.text_input("Gemini API Key", type="password")

if analysis_mode == '1 Cổ phiếu':
    symbol = st.sidebar.text_input("Mã cổ phiếu", value="VNM").upper()
    if symbol:
        data = get_financial_data(symbol, period=period)
        if data:
            tabs = st.tabs(["Dữ liệu", "Thống kê", "Trực quan", "AI Analysis"])
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
                    fig, ax = plt.subplots()
                    sns.barplot(data=df_plot, x=df_plot.index, y='NetProfit', ax=ax)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

            with tabs[3]:
                if api_key and st.button("Chạy AI"):
                    st.write(get_ai_analysis(stats_dfs, symbol, PERIOD_OPTIONS[period], api_key))

else: # Chế độ danh sách
    input_list = st.sidebar.text_area("Danh sách mã", value=", ".join(DEFAULT_STOCK_LIST))
    stock_list = [s.strip().upper() for s in input_list.replace('\n', ',').split(',') if s.strip()]
    
    if st.button(f"Tải dữ liệu cho {len(stock_list)} mã"):
        all_data = get_all_financial_data(stock_list, period)
        if all_data:
            c1, c2 = st.columns(2)
            c1.download_button("📦 Tải ZIP Excel", create_zip_file(all_data, period, 'excel'), "data.zip")
            c2.download_button("📄 Tải ZIP TXT", create_zip_file(all_data, period, 'txt'), "data_txt.zip")
