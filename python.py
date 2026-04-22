import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io

# --- 1. CẤU HÌNH HỆ THỐNG ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Financial Analyst Pro", layout="wide", page_icon="📈")

# Lấy API Key từ Streamlit Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

REPORT_TYPES = {
    'financials': 'Báo cáo Kết quả Kinh doanh',
    'balance_sheet': 'Bảng Cân đối Kế toán',
    'cashflow': 'Báo cáo Lưu chuyển Tiền tệ'
}

# --- 2. HÀM TẢI DỮ LIỆU (Dùng cache_resource để tránh lỗi Unserializable) ---
@st.cache_resource(show_spinner=False)
def get_full_stock_data(symbol):
    ticker_symbol = symbol.strip().upper()
    if len(ticker_symbol) == 3: 
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

def calculate_indicators(df):
    df = df.copy()
    # Chỉ số RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    # Đường trung bình MA
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    return df

# --- 3. TRÍ TUỆ NHÂN TẠO (GEMINI) ---
def analyze_with_ai(symbol, data_dict, info, tech_data):
    if not GEMINI_API_KEY:
        return "⚠️ Vui lòng cấu hình GEMINI_API_KEY trong Streamlit Secrets."
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        last_price = info.get('currentPrice', tech_data['Close'].iloc[-1])
        rsi_val = tech_data['RSI'].iloc[-1]
        
        context = (
            f"Phân tích cổ phiếu: {symbol} ({info.get('longName', '')})\n"
            f"Giá: {last_price:,}. RSI: {rsi_val:.2f}.\n"
            f"--- Báo cáo tài chính gần nhất ---\n"
            f"{data_dict['financials'].iloc[:, :2].to_string()}\n"
        )
        
        prompt = (
            "Bạn là một chuyên gia phân tích chứng khoán cấp cao. Dựa trên dữ liệu trên, hãy:\n"
            "1. Nhận xét sức khỏe tài chính (tăng trưởng, nợ).\n"
            "2. Phân tích xu hướng kỹ thuật ngắn hạn từ RSI/MA.\n"
            "3. Nêu 2 cơ hội và 2 rủi ro.\n"
            "4. Đưa ra chiến lược hành động (Mua/Bán/Găm giữ).\n"
            "Trả lời bằng tiếng Việt, chuyên nghiệp, định dạng Markdown rõ ràng."
        )
        
        response = client.models.generate_content(model="gemini-2.0-flash", contents=[context + prompt])
        return response.text
    except Exception as e:
        return f"❌ Lỗi AI: {str(e)}"

# --- 4. GIAO DIỆN NGƯỜI DÙNG ---
st.title("📈 AI Financial & Trading Intelligence")
st.markdown("---")

# Sidebar cấu hình
symbol = st.sidebar.text_input("Mã cổ phiếu (VD: FPT, VNM, AAPL):", value="FPT").upper()
period_choice = st.sidebar.radio("Kỳ báo cáo tài chính:", ['Năm', 'Quý'])
period_suffix = '_y' if period_choice == 'Năm' else '_q'

if symbol:
    with st.spinner(f"Đang phân tích dữ liệu cho {symbol}..."):
        data = get_full_stock_data(symbol)
    
    if data:
        info = data['info']
        hist = data['hist']
        
        # --- PHẦN 1: TỔNG QUAN (METRICS) ---
        c1, c2, c3, c4 = st.columns(4)
        curr_p = info.get('currentPrice', hist['Close'].iloc[-1])
        c1.metric("Giá hiện tại", f"{curr_p:,.0f}", f"{info.get('targetMeanPrice', 0):,.0f} (Mục tiêu)")
        c2.metric("Vốn hóa", f"{info.get('marketCap', 0)/1e12:.2f}T VNĐ")
        c3.metric("P/E", f"{info.get('trailingPE', 'N/A')}")
        c4.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%")

        # --- PHẦN 2: TABS CHỨC NĂNG ---
        t1, t2, t3, t4 = st.tabs(["📊 Biểu đồ kỹ thuật", "🧠 Trợ lý AI", "📁 Báo cáo tài chính", "📰 So sánh & Tin tức"])

        # Tab 1: Biểu đồ nến và RSI
        with t1:
            df_plot = calculate_indicators(hist)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                               row_heights=[0.7, 0.3], subplot_titles=("Giá & Đường trung bình MA", "Chỉ số sức mạnh tương đối RSI"))
            
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], 
                                       low=df_plot['Low'], close=df_plot['Close'], name="Giá"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'], name="MA20", line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'], name="MA50", line=dict(color='blue')), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name="RSI", line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            fig.update_layout(height=650, xaxis_rangeslider_visible=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        # Tab 2: Phân tích AI
        with t2:
            st.subheader("Nhận định chuyên sâu từ Gemini 2.0")
            if st.button("🚀 Chạy phân tích AI"):
                current_fin = { 'financials': data[f'financials{period_suffix}'] }
                with st.spinner("AI đang 'soi' dữ liệu..."):
                    analysis = analyze_with_ai(symbol, current_fin, info, df_plot)
                    st.markdown(analysis)
            else:
                st.info("Nhấn nút để bắt đầu phân tích kết hợp Cơ bản và Kỹ thuật.")

        # Tab 3: Báo cáo tài chính thô
        with t3:
            for key, name in REPORT_TYPES.items():
                df_report = data[f"{key}{period_suffix}"]
                with st.expander(name):
                    st.dataframe(df_report, use_container_width=True)
            
            # Xuất Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for key in REPORT_TYPES.keys():
                    data[f"{key}{period_suffix}"].to_excel(writer, sheet_name=key[:30])
            st.download_button("📥 Tải Báo cáo Excel", output.getvalue(), f"{symbol}_data.xlsx")

        # Tab 4: So sánh & Tin tức (Sửa lỗi Arrow và lỗi Title)
        with t4:
            c_left, c_right = st.columns([1, 2])
            with c_left:
                st.subheader("👥 So sánh chỉ số")
                compare_data = {
                    "Chỉ số": ["P/E", "P/B", "EV/EBITDA", "Biên LN Gộp"],
                    "Giá trị": [
                        str(info.get('trailingPE', 'N/A')), 
                        str(info.get('priceToBook', 'N/A')), 
                        str(info.get('enterpriseToEbitda', 'N/A')), 
                        f"{info.get('grossMargins', 0)*100:.2f}%"
                    ]
                }
                # Fix lỗi Arrow bằng cách ép kiểu string đồng bộ
                st.table(pd.DataFrame(compare_data).astype(str))
            
            with c_right:
                st.subheader("📰 Tin tức mới nhất")
                raw_news = data['stock'].news
                # Lọc bỏ tin rác, tin không có tiêu đề
                valid_news = [n for n in raw_news if n.get('title') and n.get('title') != "None"]

                if valid_news:
                    for item in valid_news[:5]:
                        with st.container():
                            st.markdown(f"**{item.get('title')}**")
                            cn1, cn2 = st.columns([2, 1])
                            cn1.caption(f"🆔 Nguồn: {item.get('publisher', 'Nguồn uy tín')}")
                            cn2.markdown(f"[Xem chi tiết ↗️]({item.get('link', '#')})")
                            st.divider()
                else:
                    st.warning("Không tìm thấy tin tức cụ thể cho mã này trên Yahoo Finance.")
    else:
        st.error("⚠️ Không thể tìm thấy dữ liệu cho mã này. Vui lòng kiểm tra lại (VD: VNM, FPT, AAPL).")

# Footer
st.markdown("---")
st.caption("Dữ liệu được cung cấp bởi Yahoo Finance. Phân tích bởi Gemini AI.")
