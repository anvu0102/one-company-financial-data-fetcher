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
            "3. 2 Điểm sáng & 2 Rủi ro. 4. Khuyến nghị hành động. Trả lời tiếng Việt, súc tích."
        )
        
        response = client.models.generate_content(model="gemini-2.0-flash", contents=[context + prompt])
        return response.text
    except Exception as e:
        return f"❌ Lỗi AI: {str(e)}"

# --- 4. GIAO DIỆN ---
st.title("🚀 AI Financial & Trading Intelligence")

# Sidebar
symbol = st.sidebar.text_input("Nhập mã cổ phiếu (VD: FPT, VNM, AAPL):", value="FPT").upper()
period_choice = st.sidebar.radio("Kỳ báo cáo tài chính:", ['Năm', 'Quý'])
period_suffix = '_y' if period_choice == 'Năm' else '_q'

if symbol:
    with st.spinner(f"Đang tải dữ liệu {symbol}..."):
        data = get_full_stock_data(symbol)
    
    if data:
        info = data['info']
        hist = data['hist']
        
        # --- HEADER METRICS ---
        c1, c2, c3, c4 = st.columns(4)
        curr_price = info.get('currentPrice', hist['Close'].iloc[-1])
        c1.metric("Giá hiện tại", f"{curr_price:,.0f}", f"{info.get('targetMeanPrice', 0):,.0f} (Target)")
        c2.metric("Vốn hóa", f"{info.get('marketCap', 0)/1e12:.2f}T VNĐ")
        c3.metric("P/E", f"{info.get('trailingPE', 'N/A')}")
        c4.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%")

        # --- TABS ---
        t1, t2, t3, t4 = st.tabs(["📈 Biểu đồ kỹ thuật", "🧠 AI Analysis", "📊 Báo cáo tài chính", "👥 So sánh & Tin tức"])

        with t1:
            df_plot = calculate_indicators(hist)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                               row_heights=[0.7, 0.3], subplot_titles=("Giá & MA", "RSI"))
            
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], 
                                       low=df_plot['Low'], close=df_plot['Close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], name="MA20", line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            st.subheader("Nhận định từ Trợ lý AI Gemini")
            if st.button("🚀 Chạy phân tích chiến lược"):
                current_fin = { 'financials': data[f'financials{period_suffix}'] }
                analysis = analyze_with_ai(symbol, current_fin, info, df_plot)
                st.info(analysis)

        with t3:
            for key, name in REPORT_TYPES.items():
                df_report = data[f"{key}{period_suffix}"]
                with st.expander(name):
                    st.dataframe(df_report, use_container_width=True)
            
            # Download Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for key in REPORT_TYPES.keys():
                    data[f"{key}{period_suffix}"].to_excel(writer, sheet_name=key[:30])
            st.download_button("📥 Tải Báo cáo Excel", output.getvalue(), f"{symbol}_financials.xlsx")

        with t4:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("So sánh chỉ số")
                compare_data = {
                    "Chỉ số": ["P/E", "P/B", "EV/EBITDA", "Biên LN Gộp"],
                    "Giá trị": [info.get('trailingPE', 'N/A'), info.get('priceToBook', 'N/A'), 
                               info.get('enterpriseToEbitda', 'N/A'), f"{info.get('grossMargins', 0)*100:.2f}%"]
                }
                st.table(pd.DataFrame(compare_data))
            
            with col_b:
                st.subheader("Tin tức mới nhất")
                news = data['stock'].news
                if news:
                    for item in news[:5]:
                        # SỬA LỖI KEYERROR: Sử dụng .get() để an toàn
                        title = item.get('title', 'Không có tiêu đề')
                        link = item.get('link', '#')
                        publisher = item.get('publisher', 'Nguồn không xác định')
                        with st.expander(title):
                            st.write(f"Nguồn: {publisher}")
                            st.write(f"[Đọc chi tiết tại đây]({link})")
                else:
                    st.write("Không tìm thấy tin tức mới.")
    else:
        st.error("Không tìm thấy dữ liệu. Hãy kiểm tra lại mã cổ phiếu (Ví dụ: VNM, FPT, HPG).")
