import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

start = '2014-01-01'
end = '2024-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start, end)

#Description of data
st.subheader('Data from 2014-2024')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

# Moving Average
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


#Name of the stock present in the dataset of yfinance are:
# 1. AAPL
# 2. MSFT
# 3. GOOG
# 4. AMZN
# 5. FB
# 6. TSLA
# 7. NVDA
# 8. JPM
# 9. SBI
# 10. TCS.BO
# 11. RELIANCE.NS
