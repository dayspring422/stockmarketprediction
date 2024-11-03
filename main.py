import streamlit as st
import datetime as date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# Title and introduction
st.title("Stock Price Prediction App")
st.write("""
         Welcome to the Stock Price Prediction App. You can use this app to forecast stock prices using historical data.
         """)

# Sidebar with user inputs
st.sidebar.header('User Input Parameters')

# Date range selector
START = "2010-01-01"
TODAY = date.date.today().strftime("%Y-%m-%d")

# Stock ticker selection
stocks = ("AAPL", "GOOG", "MSFT", "NVDA")
selected_stock = st.sidebar.selectbox("Select Stock Ticker", stocks)

# Number of years for prediction
n_years = st.sidebar.slider('Years of Prediction:', 1, 4)
period = n_years * 365

# Date range selection
start_date = st.sidebar.date_input('Start Date', value=date.datetime.strptime(START, "%Y-%m-%d").date())
end_date = st.sidebar.date_input('End Date', value=date.datetime.strptime(TODAY, "%Y-%m-%d").date())

# Load data function with caching
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

# Load data based on user inputs
data_load_state = st.text('Loading data...')
data = load_data(selected_stock, start_date, end_date)
data_load_state.text('Loading data...done!')

# Display raw data
st.subheader('Raw Data')
st.write(data.tail())

# Plot raw data
st.subheader('Stock Price Movement')
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name='Open'))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
fig.update_layout(title_text='Stock Price Movement Over Time', xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# Forecasting
st.subheader('Stock Price Forecasting')

# Prepare data for Prophet
df_train = data[['Date', 'Close']].copy()
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Create Prophet model
m = Prophet()
m.fit(df_train)

# Make future dataframe for prediction
future = m.make_future_dataframe(periods=period)

# Predict future values
forecast = m.predict(future)

# Plot forecasted data
fig_forecast = plot_plotly(m, forecast)
st.plotly_chart(fig_forecast)

# Display forecast components
st.subheader('Forecast Components')
fig_components = m.plot_components(forecast)
st.write(fig_components)

# Additional features and insights
st.markdown("---")
st.subheader("Additional Insights")

# Show last 5 days of forecast
st.write("Last 5 days of forecast:")
st.write(forecast.tail())

# Show uncertainty intervals
st.write("Forecast uncertainty intervals:")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# About section with creator information
st.markdown("---")
st.subheader("About")
st.write("""
         This app uses historical stock data to predict future stock prices using the Prophet forecasting model.
         """)

# Creator information
st.markdown("---")
st.write("Developed by Dayspring Idahosa")
