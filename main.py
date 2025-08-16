import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# =========================
# App Title & Intro
# =========================
st.title("Stock Price Prediction App")
st.write(
    "Forecast stock prices from historical data using Prophet. "
    "Pick a ticker and date range in the sidebar."
)

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("User Input Parameters")

START = "2010-01-01"
TODAY = dt.date.today().strftime("%Y-%m-%d")

stocks = ("AAPL", "GOOG", "MSFT", "NVDA")
selected_stock = st.sidebar.selectbox("Select Stock Ticker", stocks)

n_years = st.sidebar.slider("Years of Prediction:", 1, 4)
period = n_years * 365  # Prophet expects days

start_date = st.sidebar.date_input("Start Date", value=dt.datetime.strptime(START, "%Y-%m-%d").date())
end_date = st.sidebar.date_input("End Date", value=dt.datetime.strptime(TODAY, "%Y-%m-%d").date())

# =========================
# Data Loading
# =========================
@st.cache_data(ttl=3600)
def load_data(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download OHLCV from yfinance and return normalized DataFrame with Date column."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.reset_index()  # ensure a 'Date' column exists

def build_prophet_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DF with exactly two columns for Prophet:
      ds: datetime64[ns]
      y : float (1-D Series)
    Handles wide/multiindex data defensively.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ds", "y"])

    # Find a date-like column
    date_col = None
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("date", "datetime", "time", "timestamp"):
            date_col = c
            break
    if date_col is None:
        return pd.DataFrame(columns=["ds", "y"])

    # Find a close-like column
    close_col = None
    if "Close" in df.columns:
        close_col = "Close"
    else:
        candidates = [c for c in df.columns if str(c).lower().endswith("close")]
        if candidates:
            close_col = candidates[0]
        else:
            # handle cases like MultiIndex columns ('Close','AAPL')
            for c in df.columns:
                if isinstance(c, tuple) and len(c) >= 1 and str(c[0]).lower() == "close":
                    close_col = c
                    break
    if close_col is None:
        return pd.DataFrame(columns=["ds", "y"])

    # Extract y; squeeze to 1-D
    raw_y = df[close_col]
    if isinstance(raw_y, pd.DataFrame):
        raw_y = raw_y.iloc[:, 0]
    elif isinstance(raw_y, (np.ndarray, list, tuple)):
        raw_y = pd.Series(raw_y)

    out = pd.DataFrame({
        "ds": pd.to_datetime(df[date_col], errors="coerce"),
        "y": pd.to_numeric(pd.Series(np.asarray(raw_y).ravel()), errors="coerce")
    })
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["ds", "y"])
    out["y"] = out["y"].astype(float)

    # exactly two columns, right order
    return out[["ds", "y"]].copy()

st.text("Loading data...")
data = load_data(selected_stock, start_date, end_date)
if data.empty:
    st.error("No data returned. Check ticker or date range.")
    st.stop()
st.success("Loading data... done!")

# =========================
# Raw Data & Chart
# =========================
st.subheader("Raw Data (tail)")
st.write(data.tail())

st.subheader("Stock Price Movement")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], mode="lines", name="Open"))
fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", name="Close"))
fig.update_layout(title_text="Stock Price Movement Over Time", xaxis_rangeslider_visible=True)
st.plotly_chart(fig, use_container_width=True)

# =========================
# Forecasting
# =========================
st.subheader("Stock Price Forecasting")

df_train = build_prophet_frame(data)
if df_train.empty:
    st.error("Unable to build training set for Prophet (need valid 'ds' and numeric 'y'). Try a different date range.")
    st.stop()

# Debug preview to ensure Prophet sees correct shapes/dtypes
with st.expander("Debug: Prophet training frame preview"):
    st.write(df_train.dtypes)
    st.write(("shape:", df_train.shape))
    st.write(df_train.head())

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)  # extend by selected days
forecast = m.predict(future)

# Plotly forecast
fig_forecast = plot_plotly(m, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

# Components (matplotlib)
st.subheader("Forecast Components")
fig_components = m.plot_components(forecast)
st.pyplot(fig_components)

# =========================
# Additional Insights
# =========================
st.markdown("---")
st.subheader("Additional Insights")
st.write("Last 5 rows of forecast:")
st.write(forecast.tail())

st.write("Forecast uncertainty intervals (tail):")
st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

# =========================
# About
# =========================
st.markdown("---")
st.subheader("About")
st.write("This app uses Prophet to forecast stock prices from historical OHLC data (via yfinance).")
st.write("Developed by Dayspring Idahosa")
