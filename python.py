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

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

REPORT_TYPES = {
    'financials': 'Báo cáo Kết quả Kinh doanh',
    'balance_sheet': 'Bảng Cân đối Kế toán',
    'cashflow': 'Báo cáo Lưu chuyển Tiền tệ'
}

# --- 2. HÀM TẢI DỮ LIỆU ---
@st.cache_resource(show_spinner=False)
def get_full_stock_data(symbol):
    ticker_symbol = symbol.strip().upper()
    # Danh sách mã US để không tự động thêm .VN
    us_indices = ['AAPL', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'BTC-USD', 'NFLX']
    if len(ticker_symbol) == 3 and ticker_symbol not in us_indices: 
        ticker_symbol = f"{ticker_symbol}.VN"
    
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
        if hist.empty: return None
        return {
            "stock": stock, "hist": hist, "info": stock.info,
            "financials_y": stock.financials, "balance_sheet_y": stock.balance_sheet, "cashflow_y": stock.cashflow,
            "financials_q": stock.quarterly_financials, "balance_sheet_q": stock.quarterly_balance_sheet, "cashflow_q": stock.quarterly_cashflow
        }
    except: return None

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
    df['MA50'] = df['Close'].rolling(window=50).mean() # Đường MA50
    return df

# --- 3. AI ANALYSIS (GEMINI-3.6-FLASH) ---
def analyze_with_ai(symbol, data_dict, info, tech_data):
    if not GEMINI_API_KEY: return "⚠️ Thiếu API Key trong Secrets."
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        last_price = info.get('currentPrice', tech_data['Close'].iloc[-1])
        context = (
            f"Mã: {symbol}. Giá: {last_price:,.0f}. RSI: {tech_data['RSI'].iloc[-1]:.2f}\n"
            f"MA20: {tech_data['MA20'].iloc[-1]:.0f}, MA50: {tech_data['MA50'].iloc[-1]:.0f}\n"
            f"Tài chính:\n{data_dict['financials'].iloc[:, :2].to_string()}"
        )
        response = client.models.generate_content(
            model="gemini-3.6-flash", 
            contents=[f"Phân tích chuyên sâu cổ phiếu sau dựa trên cả kỹ thuật và cơ bản: {context}"]
        )
        return response.text
    except Exception as e: return f"❌ Lỗi AI: {str(e)}"

# --- 4. GIAO DIỆN ---
st.title("🚀 AI Stock Analytics Pro")

symbol = st.sidebar.text_input("Nhập mã cổ phiếu:", value="AAPL").upper()
period_choice = st.sidebar.radio("Kỳ báo cáo:", ['Năm', 'Quý'])
suffix = '_y' if period_choice == 'Năm' else '_q'

if symbol:
    data = get_full_stock_data(symbol)
    if data:
        info = data['info']
        hist = data['hist']
        df_plot = calculate_indicators(hist)
        
        # Header Metrics
        c1, c2, c3, c4 = st.columns(4)
        curr_p = info.get('currentPrice', hist['Close'].iloc[-1])
        c1.metric("Giá hiện tại", f"{curr_p:,.0f}", f"{info.get('trailingPE', 0):.2f} P/E")
        c2.metric("Vốn hóa", f"{info.get('marketCap', 0)/1e12:.2f}T")
        c3.metric("RSI (14)", f"{df_plot['RSI'].iloc[-1]:.1f}")
        c4.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.1f}%")

        tabs = st.tabs(["📊 Biểu đồ kỹ thuật", "🧠 Trợ lý AI", "📁 Báo cáo tài chính", "📰 Tin tức"])

        with tabs[0]:
            # Tạo Subplots: Row 1 là Giá, Row 2 là RSI
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.08, row_heights=[0.7, 0.3])
            
            # 1. Đồ thị nến (Row 1)
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], 
                                       low=df_plot['Low'], close=df_plot['Close'], name="Giá nến"), row=1, col=1)
            
            # 2. Thêm MA20 (Cam)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], name="MA20", 
                                   line=dict(color='#FF9800', width=1.5)), row=1, col=1)
            
            # 3. Thêm MA50 (Xanh dương) - Đảm bảo tính năng này hiện lên
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], name="MA50", 
                                   line=dict(color='#2196F3', width=2)), row=1, col=1)
            
            # 4. Đồ thị RSI (Row 2)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name="RSI", 
                                   line=dict(color='#9C27B0', width=2)), row=2, col=1)
            
            # 5. Thêm ngưỡng Threshold 70 và 30 cho RSI
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Overbought (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Oversold (30)")

            fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_white",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            st.subheader("🤖 Phân tích bởi Gemini 3.6 Flash")
            if st.button("🚀 Thực hiện nhận định"):
                current_fin = { 'financials': data[f'financials{suffix}'] }
                with st.spinner("AI đang đọc dữ liệu..."):
                    res = analyze_with_ai(symbol, current_fin, info, df_plot)
                    st.markdown(res)

        with tabs[2]:
            for key, name in REPORT_TYPES.items():
                with st.expander(name):
                    st.dataframe(data[f"{key}{suffix}"], use_container_width=True)
            
            # Xuất file Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for key in REPORT_TYPES.keys():
                    data[f"{key}{suffix}"].to_excel(writer, sheet_name=key[:30])
            st.download_button("📥 Tải dữ liệu Excel", output.getvalue(), f"{symbol}_financials.xlsx")

        with tabs[3]:
            st.subheader("📰 Tin tức & Định giá")
            cl, cr = st.columns([1, 2])
            with cl:
                st.write("**Chỉ số cơ bản:**")
                comp = {"Chỉ số": ["P/E", "P/B", "Mục tiêu"], "Giá trị": [str(info.get('trailingPE', 'N/A')), str(info.get('priceToBook', 'N/A')), f"{info.get('targetMeanPrice', 0):,.0f}"]}
                st.table(pd.DataFrame(comp).astype(str))
            
            with cr:
                try:
                    news = data['stock'].news
                    if news:
                        for item in news[:6]:
                            title = item.get('title') or item.get('content', {}).get('title', 'No Title')
                            link = item.get('link') or item.get('content', {}).get('clickThroughUrl', {}).get('url', '#')
                            if title != 'No Title':
                                st.markdown(f"**[{title}]({link})**")
                                st.caption(f"Nguồn: {item.get('publisher', 'Yahoo Finance')}")
                                st.divider()
                    else:
                        st.info("Không có tin tức mới cho mã này.")
                except:
                    st.warning("Không thể tải tin tức lúc này.")
    else:
        st.error("⚠️ Không tìm thấy dữ liệu cho mã cổ phiếu này.")
