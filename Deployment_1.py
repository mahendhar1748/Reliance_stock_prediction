# pip install streamlit prophet yfinance plotly

'''Prophet is a procedure for forecasting time series data based on an additive model 
where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.'''

import pandas as pd
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go



# Giving the title For Application
st.title('Stock Forecast App')

#Choosing How many years you have to Predict
n_years = st.slider('Years of prediction:', 1, 4)

#Choosing How many days you have to Predict
#n_days = st.slider('Years of prediction:', 1, 30)

# Here we are predicting for 365 days ---> we can cange if we want
#period = n_days
period = n_years * 365


def load_data(ticker):
    data = pd.read_csv('RELIANCE_2015_23.CSV')
    #data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data('RELIANCE_2015_23.CSV')
data_load_state.text("Loading data ... done")

st.subheader("Raw Data")
st.write(data.tail())
# Check for missing values
st.write(data.isna().sum())

# Handle missing values (e.g., drop rows with NaN values)
data = data.dropna()

# Ensure the DataFrame has at least two non-NaN rows
if data.shape[0] < 2:
    st.error("Dataframe has less than 2 non-NaN rows")
else:
    st.write("Dataframe is ready for forecasting")
    # Proceed with forecasting logic


# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
