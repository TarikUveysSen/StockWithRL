import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Fetch historical stock data for AAPL
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2021-12-31"
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate daily price changes
data['PriceChange'] = data['Adj Close'].diff()

# Determine if each day is an "Up" day (price increased) or a "Down" day (price decreased)
data['Trend'] = 'No Change'
data.loc[data['PriceChange'] > 0, 'Trend'] = 'Up'
data.loc[data['PriceChange'] < 0, 'Trend'] = 'Down'

# Calculate the strength of each trend
up_strength = data[data['Trend'] == 'Up']['PriceChange'].mean()
down_strength = data[data['Trend'] == 'Down']['PriceChange'].mean()

# Print the results
print("Upward Trend Strength:", up_strength)
print("Downward Trend Strength:", down_strength)

# Plot the price changes
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Adj Close'], label='AAPL Close Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('AAPL Stock Price and Trends')
plt.legend()
plt.grid(True)
plt.show()
