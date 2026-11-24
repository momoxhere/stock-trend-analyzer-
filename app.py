import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    
    # Prepare base data for Prophet
    df_prophet = data[['Close']].copy().reset_index()
    df_prophet.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
    
    # --- Model Evaluation (80/20 Chronological Split) ---
    st.subheader("Model Evaluation Metrics")
    with st.spinner('Calculating MAE and RMSE...'):
        split_idx = int(len(df_prophet) * 0.8)
        train_eval = df_prophet.iloc[:split_idx]
        test_eval = df_prophet.iloc[split_idx:]
        
        # Train on 80%
        m_eval = Prophet(daily_seasonality=True)
        m_eval.fit(train_eval)
        
        # Predict the remaining 20%
        future_eval = test_eval[['ds']]
        forecast_eval = m_eval.predict(future_eval)
        
        # Calculate Metrics
        mae = mean_absolute_error(test_eval['y'], forecast_eval['yhat'])
        rmse = np.sqrt(mean_squared_error(test_eval['y'], forecast_eval['yhat']))
        
        col1, col2 = st.columns(2)
        col1.metric("Mean Absolute Error (MAE)", f"${mae:.2f}")
        col2.metric("Root Mean Squared Error (RMSE)", f"${rmse:.2f}")
        
        with st.expander("What do these metrics mean?"):
            st.write("**MAE:** On average, the model's predictions are off by this dollar amount.")
            st.write("**RMSE:** Similar to MAE, but penalizes larger errors more heavily. A lower number is better.")

    # --- Future Forecasting (Trained on 100% of data) ---
    st.subheader("Future Price Forecast (Meta Prophet)")
    with st.spinner('Training final model on full dataset...'):
        m_final = Prophet(daily_seasonality=True)
        m_final.fit(df_prophet)
        
        period_forecast = n_months * 30
        future_final = m_final.make_future_dataframe(periods=period_forecast)
        forecast_final = m_final.predict(future_final)
        
        # Build custom Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name='Actual Price', mode='markers', marker=dict(color='black', size=4)))
        fig.add_trace(go.Scatter(x=forecast_final['ds'], y=forecast_final['yhat'], name='Predicted Trend', mode='lines', line=dict(color='#00B8D9')))
        fig.add_trace(go.Scatter(x=forecast_final['ds'], y=forecast_final['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=forecast_final['ds'], y=forecast_final['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', fillcolor='rgba(0, 184, 217, 0.2)', name='Confidence Interval'))
        
        fig.update_layout(xaxis_title="Date", yaxis_title="Price", margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("No data found. Check the ticker symbol.")