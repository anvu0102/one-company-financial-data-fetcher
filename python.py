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
    # Tự động thêm .VN cho mã 3 chữ cái Việt Nam
    us_indices = ['AAPL', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META', 'BTC-USD']
    if len(ticker_symbol) == 3 and ticker_symbol not in us_indices: 
        ticker_symbol = f"{ticker_symbol}.VN"
    
    try:
        stock = yf.Ticker(ticker_symbol)
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
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    return df

# --- 3. TRÍ TUỆ NHÂN TẠO (GEMINI-2.5-FLASH-LITE) ---
def analyze_with_ai(symbol, data_dict, info, tech_data):
    if not GEMINI_API_KEY:
        return "⚠️ Vui lòng cấu hình GEMINI_API_KEY trong Streamlit Secrets."
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        last_price = info.get('currentPrice', tech_data['Close'].iloc[-1])
        context = (
            f"Mã: {symbol} ({info.get('longName', '')})\n"
            f"Giá: {last_price:,}. RSI: {tech_data['RSI'].iloc[-1]:.2f}\n"
            f"--- Tài chính ---\n{data_dict['financials'].iloc[:, :2].to_string()}"
        )
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=[f"Phân tích tài chính chuyên sâu và đưa ra khuyến nghị cho: {context}"]
        )
        return response.text
    except Exception as e:
        return f"❌ Lỗi AI: {str(e)}"

# --- 4. GIAO DIỆN ---
st.title("🚀 AI Financial & Trading Intelligence")

symbol = st.sidebar.text_input("Mã cổ phiếu:", value="AAPL").upper()
period_choice = st.sidebar.radio("Kỳ báo cáo:", ['Năm', 'Quý'])
suffix = '_y' if period_choice == 'Năm' else '_q'

if symbol:
    data = get_full_stock_data(symbol)
    if data:
        info = data['info']
        hist = data['hist']
        
        # Header Metrics
        c1, c2, c3, c4 = st.columns(4)
        curr_p = info.get('currentPrice', hist['Close'].iloc[-1])
        c1.metric("Giá", f"{curr_p:,.0f}", f"{info.get('trailingPE', 0):.2f} PE")
        c2.metric("Vốn hóa", f"{info.get('marketCap', 0)/1e12:.2f}T")
        c3.metric("Biên LN", f"{info.get('grossMargins', 0)*100:.1f}%")
        c4.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.1f}%")

        tabs = st.tabs(["📈 Kỹ thuật", "🧠 AI Analysis", "📊 Tài chính", "📰 Tin tức & So sánh"])

        with tabs[0]:
            df_plot = calculate_indicators(hist)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], name="MA20", line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
            fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            if st.button("🚀 Phân tích với Gemini 2.5 Flash-Lite"):
                current_fin = { 'financials': data[f'financials{suffix}'] }
                res = analyze_with_ai(symbol, current_fin, info, df_plot)
                st.info(res)

        with tabs[2]:
            for key, name in REPORT_TYPES.items():
                with st.expander(name):
                    st.dataframe(data[f"{key}{suffix}"], use_container_width=True)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for key in REPORT_TYPES.keys():
                    data[f"{key}{suffix}"].to_excel(writer, sheet_name=key[:30])
            st.download_button("📥 Tải Excel", output.getvalue(), f"{symbol}_data.xlsx")

        with tabs[3]:
            col_l, col_r = st.columns([1, 2])
            with col_l:
                st.subheader("👥 So sánh")
                comp = {"Chỉ số": ["P/E", "P/B", "ROE"], "Giá trị": [str(info.get('trailingPE', 'N/A')), str(info.get('priceToBook', 'N/A')), f"{info.get('returnOnEquity', 0)*100:.2f}%"]}
                st.table(pd.DataFrame(comp).astype(str))
            
            with col_r:
                st.subheader("📰 Tin tức mới nhất")
                # FIX LỖI NONETYPE TẠI ĐÂY
                try:
                    raw_news = data['stock'].news
                    # Kiểm tra raw_news có phải danh sách không và không rỗng
                    if isinstance(raw_news, list) and len(raw_news) > 0:
                        for item in raw_news[:8]:
                            # Kiểm tra item có phải là dict không trước khi dùng .get()
                            if item and isinstance(item, dict):
                                title = item.get('title') or item.get('content', {}).get('title', 'No Title')
                                link = item.get('link') or item.get('content', {}).get('clickThroughUrl', {}).get('url', '#')
                                pub = item.get('publisher', 'Nguồn uy tín')
                                
                                if title != 'No Title':
                                    st.markdown(f"**[{title}]({link})**")
                                    st.caption(f"Nguồn: {pub}")
                                    st.divider()
                    else:
                        st.warning("⚠️ Không có tin tức mới cho mã này.")
                except Exception as e:
                    st.error(f"⚠️ Không thể tải tin tức: {str(e)}")
    else:
        st.error("Không tìm thấy dữ liệu.")
