import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go

st.title("Stock Price Trend & Forecast Analyzer")

st.sidebar.header("Settings")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
period = st.sidebar.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "max"], index=3)
ma_window = st.sidebar.slider("Moving Average Window", min_value=5, max_value=100, value=20)
n_months = st.sidebar.slider("Months of prediction:", 1, 12, value=3)

@st.cache_data
def load_data(ticker, time_period):
    stock = yf.Ticker(ticker)
    return stock.history(period=time_period)

data = load_data(ticker_symbol, period)

if not data.empty:
    st.subheader(f"{ticker_symbol} - Historical Price & {ma_window}-Day MA")
    
    if ma_window > len(data):
        st.warning(f"Warning: The {ma_window}-day moving average is larger than the available data ({len(data)} trading days).")
    
    data['SMA'] = data['Close'].rolling(window=ma_window).mean()
    st.line_chart(data[['Close', 'SMA']])
    
    # --- Forecasting with Prophet ---
    st.subheader("Price Forecast (Meta Prophet)")
    
    with st.spinner('Training model and generating forecast...'):
        # Prepare data for Prophet
        df_train = data[['Close']].copy().reset_index()
        df_train.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
        df_train['ds'] = df_train['ds'].dt.tz_localize(None)
        
        # Initialize and train Prophet model
        m = Prophet(daily_seasonality=True)
        m.fit(df_train)
        
        # Create future dataframe
        period_forecast = n_months * 30
        future = m.make_future_dataframe(periods=period_forecast)
        forecast = m.predict(future)
        
        # Build custom Plotly figure to bypass pandas assertion bug
        fig = go.Figure()
        
        # 1. Historical data points
        fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], name='Actual Price', mode='markers', marker=dict(color='black', size=4)))
        
        # 2. Forecasted trend line
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted Trend', mode='lines', line=dict(color='#00B8D9')))
        
        # 3. Confidence interval
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', fillcolor='rgba(0, 184, 217, 0.2)', name='Confidence Interval'))
        
        fig.update_layout(xaxis_title="Date", yaxis_title="Price", margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show raw forecast data components
        with st.expander("Show Forecast Data Components"):
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

else:
    st.error("No data found. Check the ticker symbol.")