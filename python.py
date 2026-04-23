import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io

# --- 1. CẤU HÌNH ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Financial Analyst Pro", layout="wide", page_icon="📈")

# Lấy API Key từ Secrets của Streamlit
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

REPORT_TYPES = {
    'financials': 'Báo cáo Kết quả Kinh doanh',
    'balance_sheet': 'Bảng Cân đối Kế toán',
    'cashflow': 'Báo cáo Lưu chuyển Tiền tệ'
}

# --- 2. HÀM TẢI DỮ LIỆU ---
@st.cache_resource(show_spinner=False)
def get_full_stock_data(symbol):
    # Lấy chính xác mã ticker người dùng nhập (không tự động thêm .VN)
    ticker_symbol = symbol.strip().upper()
    
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
        
        # Kiểm tra nếu không có dữ liệu trả về
        if hist.empty:
            return None
            
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
    except Exception:
        return None

def calculate_indicators(df):
    df = df.copy()
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    return df

# --- 3. AI ANALYSIS (GEMINI) ---
def analyze_with_ai(symbol, data_dict, info, tech_data):
    if not GEMINI_API_KEY: 
        return "⚠️ Vui lòng cấu hình GEMINI_API_KEY trong mục Secrets của Streamlit."
    
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        last_price = info.get('currentPrice', tech_data['Close'].iloc[-1])
        rsi_val = tech_data['RSI'].iloc[-1]
        
        context = (
            f"Mã: {symbol}. Giá hiện tại: {last_price:,.2f}. RSI: {rsi_val:.2f}\n"
            f"MA20: {tech_data['MA20'].iloc[-1]:.2f}, MA50: {tech_data['MA50'].iloc[-1]:.2f}\n"
            f"Dữ liệu tài chính gần nhất:\n{data_dict['financials'].iloc[:, :2].to_string()}"
        )
        
        prompt = f"Bạn là một chuyên gia phân tích tài chính cao cấp. Hãy phân tích chuyên sâu cổ phiếu sau dựa trên cả kỹ thuật và cơ bản: {context}. Đưa ra nhận định ngắn gọn, súc tích và khách quan."
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", # Cập nhật model mới nhất nếu cần
            contents=[prompt]
        )
        return response.text
    except Exception as e: 
        return f"❌ Lỗi AI: {str(e)}"

# --- 4. GIAO DIỆN NGƯỜI DÙNG ---
st.title("🚀 AI Stock Analytics Pro")

# Sidebar cấu hình
st.sidebar.header("Cấu hình truy vấn")
symbol = st.sidebar.text_input(
    "Nhập mã ticker:", 
    value="AAPL", 
    help="Ví dụ: AAPL (Mỹ), VNM.VN (Việt Nam), BTC-USD (Crypto)"
).upper()

period_choice = st.sidebar.radio("Kỳ báo cáo tài chính:", ['Năm', 'Quý'])
suffix = '_y' if period_choice == 'Năm' else '_q'

if symbol:
    with st.spinner(f"Đang tải dữ liệu cho {symbol}..."):
        data = get_full_stock_data(symbol)
    
    if data:
        info = data['info']
        hist = data['hist']
        df_plot = calculate_indicators(hist)
        
        # --- Hiển thị Chỉ số chính ---
        c1, c2, c3, c4 = st.columns(4)
        curr_p = info.get('currentPrice', hist['Close'].iloc[-1])
        currency = info.get('currency', 'USD')
        
        c1.metric("Giá hiện tại", f"{curr_p:,.2f} {currency}", f"{info.get('trailingPE', 0):.2f} P/E")
        c2.metric("Vốn hóa", f"{info.get('marketCap', 0)/1e9:.2f}B")
        c3.metric("RSI (14)", f"{df_plot['RSI'].iloc[-1]:.1f}")
        c4.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.1f}%")

        tabs = st.tabs(["📊 Biểu đồ kỹ thuật", "🧠 Trợ lý AI", "📁 Báo cáo tài chính", "📰 Tin tức"])

        # Tab 1: Charting
        with tabs[0]:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.08, row_heights=[0.7, 0.3])
            
            # Candlestick
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], 
                                       low=df_plot['Low'], close=df_plot['Close'], name="Giá"), row=1, col=1)
            
            # Moving Averages
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], name="MA20", 
                                   line=dict(color='#FF9800', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], name="MA50", 
                                   line=dict(color='#2196F3', width=2)), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name="RSI", 
                                   line=dict(color='#9C27B0', width=2)), row=2, col=1)
            
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_white",
                              legend=dict(orientation="h", y=1.02, x=1))
            st.plotly_chart(fig, use_container_width=True)

        # Tab 2: AI Analysis
        with tabs[1]:
            st.subheader(f"🤖 Phân tích chuyên sâu cho {symbol}")
            if st.button("🚀 Chạy phân tích AI"):
                current_fin = { 'financials': data[f'financials{suffix}'] }
                with st.spinner("AI đang xử lý dữ liệu thực tế..."):
                    res = analyze_with_ai(symbol, current_fin, info, df_plot)
                    st.markdown(res)

        # Tab 3: Financials
        with tabs[2]:
            for key, name in REPORT_TYPES.items():
                with st.expander(name):
                    st.dataframe(data[f"{key}{suffix}"], use_container_width=True)
            
            # Export to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for key in REPORT_TYPES.keys():
                    data[f"{key}{suffix}"].to_excel(writer, sheet_name=key[:30])
            st.download_button("📥 Tải dữ liệu Excel", output.getvalue(), f"{symbol}_financials.xlsx")

        # Tab 4: News
        with tabs[3]:
            st.subheader("📰 Tin tức mới nhất")
            cl, cr = st.columns([1, 2])
            with cl:
                st.write("**Định giá & Mục tiêu:**")
                comp = {
                    "Chỉ số": ["P/E", "P/B", "Target Price"], 
                    "Giá trị": [
                        str(info.get('trailingPE', 'N/A')), 
                        str(info.get('priceToBook', 'N/A')), 
                        f"{info.get('targetMeanPrice', 0):,.2f}"
                    ]
                }
                st.table(pd.DataFrame(comp))
            
            with cr:
                try:
                    news = data['stock'].news
                    if news:
                        for item in news[:6]:
                            title = item.get('title')
                            link = item.get('link')
                            publisher = item.get('publisher', 'Finance Source')
                            st.markdown(f"**[{title}]({link})**")
                            st.caption(f"Nguồn: {publisher}")
                            st.divider()
                    else:
                        st.info("Không tìm thấy tin tức cho mã này.")
                except:
                    st.warning("Không thể tải tin tức.")
    else:
        st.error(f"⚠️ Không tìm thấy dữ liệu cho mã '{symbol}'. Vui lòng kiểm tra lại ký hiệu ticker (Ví dụ: AAPL cho Apple, VNM.VN cho Vinamilk).")
