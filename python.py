import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import warnings
import io

# --- 1. IMPORT GENAI ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("Lỗi: Thư viện 'google-genai' chưa được cài đặt.")
    st.stop()

# --- 2. CẤU HÌNH ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Financial Analyst Pro", layout="wide")

REPORT_TYPES = {
    'financials': 'Báo cáo Kết quả Kinh doanh',
    'balance_sheet': 'Bảng Cân đối Kế toán',
    'cashflow': 'Báo cáo Lưu chuyển Tiền tệ'
}

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# --- 3. HÀM XỬ LÝ DỮ LIỆU & AI ---
def analyze_with_ai(symbol, data_dict, info):
    if not GEMINI_API_KEY:
        return "⚠️ Vui lòng cấu hình GEMINI_API_KEY trong Secrets."
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Tóm tắt thông tin gửi cho AI
        summary_text = f"Cổ phiếu: {symbol} ({info.get('longName', '')})\n"
        summary_text += f"Ngành: {info.get('sector', 'N/A')}\n"
        
        for key, name in REPORT_TYPES.items():
            df_summary = data_dict[key].iloc[:, :3] # Lấy 3 kỳ gần nhất
            summary_text += f"\n--- {name} ---\n{df_summary.to_string()}\n"

        prompt = (
            "Bạn là một chuyên gia phân tích tài chính cao cấp (CFA). "
            "Dựa trên dữ liệu tài chính 3 kỳ gần nhất trên, hãy thực hiện:\n"
            "1. Phân tích xu hướng doanh thu và lợi nhuận.\n"
            "2. Đánh giá sức khỏe bảng cân đối kế toán.\n"
            "3. Nêu 2 điểm tích cực và 2 rủi ro cần lưu ý.\n"
            "4. Kết luận ngắn gọn về tiềm năng.\n"
            "Trả lời bằng tiếng Việt, chuyên nghiệp, dùng các thuật ngữ tài chính chuẩn xác."
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[summary_text + prompt]
        )
        return response.text
    except Exception as e:
        return f"❌ Lỗi AI: {str(e)}"

@st.cache_data(show_spinner=False)
def get_full_data(symbol):
    ticker_symbol = symbol.strip().upper()
    if len(ticker_symbol) == 3: ticker_symbol = f"{ticker_symbol}.VN"
    
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        data = {
            'info': info,
            'yearly': {
                'financials': stock.financials,
                'balance_sheet': stock.balance_sheet,
                'cashflow': stock.cashflow
            },
            'quarterly': {
                'financials': stock.quarterly_financials,
                'balance_sheet': stock.quarterly_balance_sheet,
                'cashflow': stock.quarterly_cashflow
            }
        }
        return data if data['yearly']['financials'] is not None else None
    except:
        return None

# --- 4. GIAO DIỆN CHÍNH ---
st.title("💹 AI Financial Analyst Pro")
st.markdown("---")

# Sidebar
symbol = st.sidebar.text_input("Mã cổ phiếu (VD: VNM, FPT, AAPL):", value="VNM").upper()
period_choice = st.sidebar.selectbox("Kỳ báo cáo:", ['Năm', 'Quý'])
period_key = 'yearly' if period_choice == 'Năm' else 'quarterly'

if symbol:
    with st.spinner(f"Đang phân tích {symbol}..."):
        all_data = get_full_data(symbol)
        
    if all_data:
        info = all_data['info']
        data = all_data[period_key]
        
        # --- HEADER: THÔNG TIN CHUNG ---
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.metric("Tên công ty", info.get('longName', 'N/A'))
            st.caption(f"Ngành: {info.get('sector', 'N/A')} | Quốc gia: {info.get('country', 'N/A')}")
        with col2:
            st.metric("Giá hiện tại", f"{info.get('currentPrice', 0):,}", info.get('currency', ''))
        with col3:
            st.metric("Vốn hóa", f"{info.get('marketCap', 0) // 10**9:,} tỷ")
        with col4:
            st.metric("P/E", info.get('trailingPE', 'N/A'))

        st.markdown("---")

        # --- TAB PHÂN TÍCH ---
        t1, t2, t3 = st.tabs(["📊 Biểu đồ & Chỉ số", "🧠 AI Analysis", "📁 Dữ liệu thô"])

        with t1:
            # Vẽ biểu đồ doanh thu & lợi nhuận
            try:
                rev_df = data['financials'].loc['Total Revenue'].iloc[::-1]
                ni_df = data['financials'].loc['Net Income'].iloc[::-1]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=rev_df.index.astype(str), y=rev_df.values, name='Doanh thu'))
                fig.add_trace(go.Scatter(x=ni_df.index.astype(str), y=ni_df.values, name='LN ròng', line=dict(color='orange', width=3)))
                fig.update_layout(title="Xu hướng Doanh thu & Lợi nhuận", barmode='group', height=400)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Không đủ dữ liệu để vẽ biểu đồ.")

            # Tính toán một số chỉ số nhanh (Key Metrics)
            st.subheader("Chỉ số tài chính cơ bản")
            m1, m2, m3, m4 = st.columns(4)
            try:
                latest_fin = data['financials'].iloc[:, 0]
                latest_bs = data['balance_sheet'].iloc[:, 0]
                
                roe = (latest_fin.get('Net Income', 0) / latest_bs.get('Stockholders Equity', 1)) * 100
                margin = (latest_fin.get('Net Income', 0) / latest_fin.get('Total Revenue', 1)) * 100
                cur_ratio = latest_bs.get('Total Current Assets', 0) / latest_bs.get('Total Current Liabilities', 1)
                
                m1.metric("ROE", f"{roe:.2f}%")
                m2.metric("Biên LN ròng", f"{margin:.2f}%")
                m3.metric("Khả năng thanh toán", f"{cur_ratio:.2f}x")
                m4.metric("Nợ/Vốn CSH", f"{(latest_bs.get('Total Liabilities Net Minority Interest', 0) / latest_bs.get('Stockholders Equity', 1)):.2f}")
            except:
                st.write("Dữ liệu chỉ số không khả dụng.")

        with t2:
            st.subheader("Nhận định chuyên sâu từ Gemini AI")
            if st.button("🚀 Chạy phân tích AI ngay"):
                with st.spinner("AI đang 'soi' báo cáo..."):
                    analysis = analyze_with_ai(symbol, data, info)
                    st.markdown(f"> {analysis}")
            else:
                st.info("Nhấn nút phía trên để AI bắt đầu phân tích dữ liệu.")

        with t3:
            for key, name in REPORT_TYPES.items():
                with st.expander(name):
                    st.dataframe(data[key], use_container_width=True)
            
            # Export Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for key in REPORT_TYPES.keys():
                    data[key].to_excel(writer, sheet_name=key[:30])
            st.download_button("📥 Tải dữ liệu Excel", output.getvalue(), f"{symbol}_report.xlsx")

    else:
        st.error(f"Không tìm thấy dữ liệu cho mã '{symbol}'. Nếu là chứng khoán Việt Nam, hãy đảm bảo mã có 3 chữ cái.")

# --- 5. FOOTER ---
st.markdown("---")
st.caption("Dữ liệu cung cấp bởi Yahoo Finance. Phân tích bởi Google Gemini AI.")
