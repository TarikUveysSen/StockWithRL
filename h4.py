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

# Plot a histogram of price changes
plt.figure(figsize=(10, 6))
plt.hist(data['PriceChange'], bins=50, color='blue', alpha=0.7)
plt.xlabel('Price Change')
plt.ylabel('Frequency')
plt.title('Distribution of Price Changes for AAPL')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(data['PriceChange'], bins=np.arange(-0.5, 0.6, 0.1), color='blue', alpha=0.7)
plt.xlabel('Price Change')
plt.ylabel('Frequency')
plt.title('Distribution of Price Changes for AAPL')
plt.xticks(np.arange(-0.5, 0.6, 0.1))  # Adjust x-axis tick values
plt.grid(True)

# Highlight the bins around 0
for patch, bin_value in zip(patches, bins):
    if bin_value >= -threshold and bin_value <= threshold:
        patch.set_facecolor('red')

plt.show()

# Calculate Z-scores for both "Price Change" and "Adj Close"
z_scores_price_change = stats.zscore(df1['Price Change'])
z_scores_adj_close = stats.zscore(df1['Adj Close'])

# Create a histogram for Z-scores of Price Change centered around 0
plt.figure(figsize=(12, 6))
n, bins, patches = plt.hist(z_scores_price_change, bins=np.arange(-5, 5.5, 0.5), color='blue', alpha=0.7)
plt.xlabel('Z Score (Price Change)')
plt.ylabel('Frequency')
plt.title('Distribution of Z Scores for Price Changes (AAPL)')
plt.xticks(np.arange(-5, 6, 1))  # Adjust x-axis tick values
plt.grid(True)

# Highlight the bins around 0 for Price Change
for patch, bin_value in zip(patches, bins):
    if bin_value >= -1 and bin_value <= 1:
        patch.set_facecolor('red')

plt.show()

# Create a histogram for Z-scores of Adj Close centered around 0
plt.figure(figsize=(12, 6))
n, bins, patches = plt.hist(z_scores_adj_close, bins=np.arange(-5, 5.5, 0.5), color='green', alpha=0.7)
plt.xlabel('Z Score (Adj Close)')
plt.ylabel('Frequency')
plt.title('Distribution of Z Scores for Adj Close (AAPL)')
plt.xticks(np.arange(-5, 6, 1))  # Adjust x-axis tick values
plt.grid(True)

# Highlight the bins around 0 for Adj Close
for patch, bin_value in zip(patches, bins):
    if bin_value >= -1 and bin_value <= 1:
        patch.set_facecolor('orange')

plt.show()



import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Fetch historical stock data for AAPL
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2021-12-31"
df1 = yf.download(ticker, start=start_date, end=end_date)

# Calculate daily price changes
df1['Price Change'] = df1['Adj Close'].diff()

# Calculate daily trading volume changes
df1['Volume Change'] = df1['Volume'].diff()

# Calculate daily volatility (standard deviation of price changes)
df1['Volatility'] = df1['Price Change'].rolling(window=14).std()

# Define a threshold for categorizing "No Change"
threshold = 0.10  # 10%

# Categorize each day as Up, Down, or No Change based on Price Change
df1['Price Trend'] = 'No Change'
df1.loc[df1['Price Change'] > threshold, 'Price Trend'] = 'Up'
df1.loc[df1['Price Change'] < -threshold, 'Price Trend'] = 'Down'

# Calculate Z-scores for Price Change
z_scores_price_change = stats.zscore(df1['Price Change'])

# Calculate the Sharpe Ratio for daily returns
risk_free_rate = 0.03  # Assume a 3% annual risk-free rate
daily_returns = df1['Adj Close'].pct_change()
sharpe_ratio = (daily_returns.mean() - risk_free_rate) / daily_returns.std()

# Calculate the Sortino Ratio for daily returns
downside_returns = daily_returns[daily_returns < 0]
downside_std_dev = downside_returns.std()
sortino_ratio = (daily_returns.mean() - risk_free_rate) / downside_std_dev

# Create a histogram for Z-scores of Price Change centered around 0
plt.figure(figsize=(12, 6))
n, bins, patches = plt.hist(z_scores_price_change, bins=np.arange(-5, 5.5, 0.5), color='blue', alpha=0.7)
plt.xlabel('Z Score (Price Change)')
plt.ylabel('Frequency')
plt.title('Distribution of Z Scores for Price Changes (AAPL)')
plt.xticks(np.arange(-5, 6, 1))  # Adjust x-axis tick values
plt.grid(True)

# Highlight the bins around 0 for Price Change
for patch, bin_value in zip(patches, bins):
    if bin_value >= -1 and bin_value <= 1:
        patch.set_facecolor('red')

plt.show()

# Plot daily volatility
plt.figure(figsize=(12, 6))
plt.plot(df1.index, df1['Volatility'], label='Volatility', color='orange')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title('14-day Rolling Volatility for AAPL')
plt.legend()
plt.grid(True)
plt.show()

# Print the Sharpe Ratio and Sortino Ratio with explanations
print("Sharpe Ratio (Risk-Adjusted Return):", sharpe_ratio)
print("Sortino Ratio (Downside Risk-Adjusted Return):", sortino_ratio)

