import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import warnings

warnings.filterwarnings('ignore')

# Function to load data
def load_data(ticker):
    #data = yf.download(ticker, start='2015-01-01', end='2024-01-01')
    data = pd.read_csv('RELIANCE_2015_23.CSV')
    data.reset_index(inplace=True)
    return data

# Function to plot data
def plot_data(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Date'], data['Close'], label='Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    st.pyplot(fig)

# Function to fit ARIMA model and forecast
def forecast(data):
    # Ensure there are no missing values
    data = data.dropna()
    
    # Check if we have enough data
    if data.shape[0] < 2:
        st.error("Dataframe has less than 2 non-NaN rows")
        return
    
    # Fit ARIMA model
    model = ARIMA(data['Close'], order=(5, 1, 0))  # Order can be adjusted
    model_fit = model.fit(disp=0)
    
    # Forecast
    forecast, stderr, conf = model_fit.forecast(steps=30)  # Forecasting 30 days
    
    # Create DataFrame for forecasted data
    forecast_dates = pd.date_range(start=data['Date'].iloc[-1], periods=30, freq='D')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
    
    # Plot forecast
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Date'], data['Close'], label='Close Price')
    ax.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    st.pyplot(fig)

# Streamlit app
st.title('Reliance Stock Price Forecasting')

# Load data
ticker = 'RELIANCE_2015_23.CSV'  # Reliance Industries Limited ticker
data = load_data(ticker)

# Display data
st.subheader('Stock Price Data')
st.write(data.tail())

# Plot data
st.subheader('Stock Price Chart')
plot_data(data)

# Forecast
st.subheader('Stock Price Forecast')
forecast(data)
