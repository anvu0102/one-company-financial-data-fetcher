import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io

# --- 1. CẤU HÌNH ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Financial Analyst Pro", layout="wide", page_icon="🍎")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# --- 2. HÀM TẢI DỮ LIỆU ---
@st.cache_resource(show_spinner=False)
def get_full_stock_data(symbol):
    ticker_symbol = symbol.strip().upper()
    # Tự động thêm .VN nếu là mã 3 chữ cái (trừ các mã phổ biến US)
    us_stocks = ['AAPL', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'META']
    if len(ticker_symbol) == 3 and ticker_symbol not in us_stocks: 
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
    except Exception:
        return None

# --- 3. TRÍ TUỆ NHÂN TẠO (GEMINI-2.5-FLASH-LITE) ---
def analyze_with_ai(symbol, data_dict, info, hist_data):
    if not GEMINI_API_KEY:
        return "⚠️ Thiếu GEMINI_API_KEY trong Secrets."
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Chuẩn bị ngữ cảnh
        last_close = hist_data['Close'].iloc[-1]
        context = (
            f"Cổ phiếu: {symbol}\nGiá đóng cửa gần nhất: {last_close:,.2f}\n"
            f"Chỉ số P/E: {info.get('trailingPE', 'N/A')}\n"
            f"Doanh thu 2 kỳ gần nhất:\n{data_dict['financials'].iloc[:, :2].to_string()}"
        )
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", # Cập nhật model theo yêu cầu
            contents=[f"Bạn là chuyên gia tài chính. Hãy phân tích ngắn gọn: {context}"]
        )
        return response.text
    except Exception as e:
        return f"❌ Lỗi AI: {str(e)}"

# --- 4. GIAO DIỆN ---
st.title("🍎 Apple & Global Stock AI Analyst")

symbol = st.sidebar.text_input("Nhập mã (AAPL, FPT, TSLA...):", value="AAPL").upper()
period_choice = st.sidebar.radio("Kỳ báo cáo:", ['Năm', 'Quý'])
suffix = '_y' if period_choice == 'Năm' else '_q'

if symbol:
    data = get_full_stock_data(symbol)
    if data:
        info = data['info']
        hist = data['hist']
        
        # Tab chính
        t1, t2, t3, t4 = st.tabs(["📈 Kỹ thuật", "🧠 Phân tích AI", "📊 Tài chính", "📰 Tin tức"])

        with t1:
            # Vẽ biểu đồ đơn giản
            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], 
                            high=hist['High'], low=hist['Low'], close=hist['Close'])])
            fig.update_layout(title=f"Diễn biến giá {symbol}", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        with t2:
            if st.button("🚀 Chạy Gemini 2.5 Flash-Lite"):
                res = analyze_with_ai(symbol, {"financials": data[f"financials{suffix}"]}, info, hist)
                st.write(res)

        with t3:
            st.dataframe(data[f"financials{suffix}"], use_container_width=True)

        with t4:
            st.subheader(f"Tin tức mới nhất về {symbol}")
            try:
                # Cách lấy tin tức mới nhất từ yfinance
                news_list = data['stock'].get_news() # Sử dụng get_news() thay vì .news
                
                if news_list and len(news_list) > 0:
                    for item in news_list[:10]: # Hiển thị 10 tin
                        # Xử lý đa dạng cấu trúc dict của Yahoo
                        title = item.get('title') or item.get('content', {}).get('title', 'No Title')
                        link = item.get('link') or item.get('content', {}).get('clickThroughUrl', {}).get('url', '#')
                        pub = item.get('publisher') or item.get('content', {}).get('pubDate', '')
                        
                        if title != 'No Title':
                            with st.container():
                                st.markdown(f"**[{title}]({link})**")
                                st.caption(f"Nguồn/Ngày: {pub}")
                                st.divider()
                else:
                    st.info("💡 Không tìm thấy tin tức trực tiếp. Yahoo Finance đôi khi chặn các truy vấn tự động từ Server. Bạn có thể thử lại sau vài phút.")
            except Exception as e:
                st.error(f"Lỗi khi lấy tin tức: {e}")
    else:
        st.error("Không tìm thấy dữ liệu cho mã này.")
