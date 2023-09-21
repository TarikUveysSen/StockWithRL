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
