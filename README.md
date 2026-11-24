# Stock Price Trend Analyzer

A machine learning pipeline for fetching historical stock market data, engineering technical indicators, and preparing sequential datasets for time-series price forecasting using PyTorch LSTMs.

## Features
- **Automated Data Acquisition:** Pulls historical OHLCV market data for global and Indian stocks (e.g., `AAPL`, `RELIANCE.NS`) via `yfinance`.
- **Technical Feature Engineering:** Computes 20-day and 50-day Simple Moving Averages (SMA) alongside the 14-day Relative Strength Index (RSI) to capture trend direction and momentum.
- **Trend Visualization:** Plots closing prices overlaid with fast and slow moving averages using `matplotlib`.
- **Deep Learning Pipeline:** Scales multi-feature datasets using `MinMaxScaler`, constructs 60-day sliding time-series windows, and converts data into GPU-ready PyTorch tensors (CUDA-enabled).

## Tech Stack
- **Language:** Python
- **Data & Visualization:** Pandas, NumPy, Matplotlib, yfinance
- **Machine Learning:** PyTorch, Scikit-Learn

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install yfinance pandas matplotlib scikit-learn torch numpy
   ```

2. **Run the analyzer:**
   ```bash
   python trend_analyzer.py
   ```

## Roadmap
- [x] Historical data extraction & baseline plotting
- [x] Technical indicator engineering (SMA, RSI)
- [x] Sequence generation & PyTorch tensor preparation
- [ ] LSTM model architecture & training loop
- [ ] Evaluation metrics (MAE / RMSE) & prediction visualization
