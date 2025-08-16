import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# -----------------------------
# Title and intro
# -----------------------------
st.title("Stock Price Prediction App")
st.write(
    "Welcome to the Stock Price Prediction App. "
    "Use this app to forecast stock prices from historical data."
)

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("User Input Parameters")

START = "2010-01-01"
TODAY = dt.date.today().strftime("%Y-%m-%d")

stocks = ("AAPL", "GOOG", "MSFT", "NVDA")
selected_stock = st.sidebar.selectbox("Select Stock Ticker", stocks)

n_years = st.sidebar.slider("Years of Prediction:", 1, 4)
period = n_years * 365  # Prophet uses days

start_date = st.sidebar.date_input("Start Date", value=dt.datetime.strptime(START, "%Y-%m-%d").date())
end_date = st.sidebar.date_input("End Date", value=dt.datetime.strptime(TODAY, "%Y-%m-%d").date())

# -----------------------------
# Data loading + cleaning
# -----------------------------
@st.cache_data(ttl=3600)
def load_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download OHLCV from yfinance and return a normalized DataFrame with Date as a column."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()  # ensure Date column exists
    return df

def build_prophet_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a clean Prophet dataframe with columns:
      ds (datetime64[ns]) and y (float)
    Handles any non-numeric values and drops NaNs/infs.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ds", "y"])

    # Guard for expected columns
    if "Date" not in df.columns:
        # Some versions might name it "Datetime" or similar; try to recover
        date_col = None
        for c in df.columns:
            if str(c).lower() in ("date", "datetime", "time", "timestamp"):
                date_col = c
                break
        if date_col is None:
            return pd.DataFrame(columns=["ds", "y"])
        df = df.rename(columns={date_col: "Date"})

    if "Close" not in df.columns:
        # Defensive: if yfinance returned MultiIndex or unusual shape, try to extract Close
        # (In your current app with single ticker, 'Close' should be present.)
        possible_close = [c for c in df.columns if str(c).lower().endswith("close")]
        if not possible_close:
            return pd.DataFrame(columns=["ds", "y"])
        df = df.rename(columns={possible_close[0]: "Close"})

    out = pd.DataFrame({
        "ds": pd.to_datetime(df["Date"], errors="coerce"),
        "y": pd.to_numeric(df["Close"], errors="coerce")
    })
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["ds", "y"])
    # Ensure 1-D float Series (prevents Prophet casting error)
    out["y"] = out["y"].astype(float)
    return out[["ds", "y"]].copy()

# Load
st.text("Loading data...")
data = load_data(selected_stock, start_date, end_date)
if data.empty:
    st.error("No data returned. Check the ticker or date range.")
    st.stop()
st.success("Loading data... done!")

# -----------------------------
# Raw data display
# -----------------------------
st.subheader("Raw Data")
st.write(data.tail())

# -----------------------------
# Price chart (Open vs Close)
# -----------------------------
st.subheader("Stock Price Movement")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], mode="lines", name="Open"))
fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Close"))
fig.update_layout(title_text="Stock Price Movement Over Time", xaxis_rangeslider_visible=True)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Forecasting
# -----------------------------
st.subheader("Stock Price Forecasting")

df_train = build_prophet_frame(data)
if df_train.empty:
    st.error("Unable to build a valid training set for Prophet (ds/y). Try a different date range.")
    st.stop()

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)  # daily periods
forecast = m.predict(future)

# Plotly forecast
fig_forecast = plot_plotly(m, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

# Components (matplotlib figure)
st.subheader("Forecast Components")
fig_components = m.plot_components(forecast)
st.pyplot(fig_components)

# -----------------------------
# Additional insights
# -----------------------------
st.markdown("---")
st.subheader("Additional Insights")

st.write("Last 5 days of forecast:")
st.write(forecast.tail())

st.write("Forecast uncertainty intervals:")
st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

# -----------------------------
# About
# -----------------------------
st.markdown("---")
st.subheader("About")
st.write("This app uses historical stock data to predict future prices using the Prophet forecasting model.")

st.markdown("---")
st.write("Developed by Dayspring Idahosa")
