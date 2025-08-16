# main.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# ---------------------------------------
# Sidebar Inputs
# ---------------------------------------
st.title("Stock Price Prediction App")
st.write("Forecast stock prices from historical data using Prophet.")

st.sidebar.header("Parameters")
START = "2010-01-01"
TODAY = dt.date.today()

stocks = ("AAPL", "GOOG", "MSFT", "NVDA")
selected_stock = st.sidebar.selectbox("Select Stock Ticker", stocks)

n_years = st.sidebar.slider("Years of Prediction", 1, 4, value=2)
period_days = n_years * 365  # Prophet uses days

start_date = st.sidebar.date_input("Start Date", value=dt.datetime.strptime(START, "%Y-%m-%d").date())
end_date = st.sidebar.date_input("End Date", value=TODAY)

# Ensure valid date range
if start_date >= end_date:
    st.error("Start Date must be before End Date.")
    st.stop()

# ---------------------------------------
# Data Loading (robust) + Caching
# ---------------------------------------
@st.cache_data(ttl=3600)
def load_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Download OHLCV from yfinance for a single ticker.
    Returns a DataFrame with a 'Date' column.
    Falls back to a long period if the explicit range is empty.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        # fallback to a safe historical period
        df = yf.download(ticker, period="10y", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.reset_index()  # ensure 'Date' column exists

def build_prophet_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a clean DataFrame for Prophet with columns:
      ds: datetime64[ns]
      y : float (1-D Series)
    Handles alt date names, Adj Close, multiindex/wide frames, NaNs/Infs.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ds", "y"])

    # 1) Find a date-like column
    date_col = None
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("date", "datetime", "time", "timestamp"):
            date_col = c
            break
    if date_col is None:
        # try to auto-detect any datetime-like column
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_col = c
                break
    if date_col is None:
        return pd.DataFrame(columns=["ds", "y"])

    # 2) Choose a price column (prefer Adj Close, then Close, else anything ending with 'close')
    close_col = None
    for cand in ("Adj Close", "Close"):
        if cand in df.columns:
            close_col = cand
            break
    if close_col is None:
        candidates = [c for c in df.columns if str(c).lower().endswith("close")]
        if candidates:
            close_col = candidates[0]
        else:
            # handle MultiIndex like ('Close','AAPL')
            for c in df.columns:
                if isinstance(c, tuple) and len(c) >= 1 and str(c[0]).lower() in ("adj close", "close"):
                    close_col = c
                    break
    if close_col is None:
        return pd.DataFrame(columns=["ds", "y"])

    # 3) Extract y and force it to 1-D numeric Series
    raw_y = df[close_col]
    if isinstance(raw_y, pd.DataFrame):
        raw_y = raw_y.iloc[:, 0]  # first ticker if wide
    elif isinstance(raw_y, (np.ndarray, list, tuple)):
        raw_y = pd.Series(raw_y)

    out = pd.DataFrame({
        "ds": pd.to_datetime(df[date_col], errors="coerce"),
        "y": pd.to_numeric(pd.Series(np.asarray(raw_y).ravel()), errors="coerce")
    })

    # Clean: drop NaNs/Inf, sort by date, ensure unique ds
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["ds", "y"])
    if out.empty:
        return out
    out = out.sort_values("ds").drop_duplicates(subset=["ds"], keep="last")
    out["y"] = out["y"].astype(float)

    # Final shape/order
    return out[["ds", "y"]].copy()

# ---------------------------------------
# Load + Show Data
# ---------------------------------------
with st.spinner("Loading data..."):
    data = load_data(selected_stock, start_date, end_date)

if data.empty:
    st.error("No data returned. Try a different date range or ticker.")
    st.stop()

st.subheader("Raw Data (tail)")
st.dataframe(data.tail(), use_container_width=True)

st.subheader("Stock Price Movement")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=data["Date"], y=data.get("Open", pd.Series(index=data.index)), mode="lines", name="Open"))
fig_price.add_trace(go.Scatter(x=data["Date"], y=data.get("Close", data.get("Adj Close", pd.Series(index=data.index))), mode="lines", name="Close/Adj Close"))
fig_price.update_layout(title_text=f"{selected_stock} Price Over Time", xaxis_rangeslider_visible=True)
st.plotly_chart(fig_price, use_container_width=True)

# ---------------------------------------
# Build Prophet Frame
# ---------------------------------------
st.subheader("Stock Price Forecasting")

df_train = build_prophet_frame(data)
if df_train.empty:
    st.error("Unable to build training set for Prophet (need valid dates and numeric Close/Adj Close). Try a different date range.")
    st.stop()

# Optional debug (expand if needed)
with st.expander("Debug: Prophet training frame (ds/y)"):
    st.write(df_train.dtypes)
    st.write(("shape:", df_train.shape))
    st.write(df_train.head())

# ---------------------------------------
# Train Prophet & Forecast
# ---------------------------------------
m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period_days)  # daily periods
forecast = m.predict(future)

# Plot forecast (Plotly)
fig_forecast = plot_plotly(m, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

# Components (matplotlib)
st.subheader("Forecast Components")
fig_components = m.plot_components(forecast)
st.pyplot(fig_components)

# ---------------------------------------
# Additional Insights
# ---------------------------------------
st.markdown("---")
st.subheader("Additional Insights")
st.write("Last 5 rows of forecast:")
st.dataframe(forecast.tail(), use_container_width=True)

st.write("Forecast uncertainty intervals (tail):")
st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(), use_container_width=True)

st.markdown("---")
st.caption("Developed by Dayspring Idahosa")
