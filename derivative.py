import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your DataFrame with "Adj Close" data
# Replace 'your_dataframe.csv' with your actual DataFrame
# You may also need to adjust the date column name accordingly
df = pd.read_csv('your_dataframe.csv')
df['Date'] = pd.to_datetime(df['Date'])  # Convert date column to datetime format
df.set_index('Date', inplace=True)  # Set the date column as the index

# Calculate the first derivative (rate of change)
df['First_Derivative'] = df['Adj Close'].diff()

# Calculate the second derivative (acceleration)
df['Second_Derivative'] = df['First_Derivative'].diff()

# Calculate moving averages (MA) for both derivatives (10-day and 50-day)
df['First_Derivative_10_MA'] = df['First_Derivative'].rolling(window=10).mean()
df['First_Derivative_50_MA'] = df['First_Derivative'].rolling(window=50).mean()
df['Second_Derivative_10_MA'] = df['Second_Derivative'].rolling(window=10).mean()
df['Second_Derivative_50_MA'] = df['Second_Derivative'].rolling(window=50).mean()

# Plot the first and second derivatives and their MAs over time
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['First_Derivative'], label='First Derivative', color='blue')
plt.plot(df.index, df['Second_Derivative'], label='Second Derivative', color='green')
plt.plot(df.index, df['First_Derivative_10_MA'], label='First Derivative 10-day MA', linestyle='--', color='orange')
plt.plot(df.index, df['First_Derivative_50_MA'], label='First Derivative 50-day MA', linestyle='--', color='red')
plt.plot(df.index, df['Second_Derivative_10_MA'], label='Second Derivative 10-day MA', linestyle='--', color='purple')
plt.plot(df.index, df['Second_Derivative_50_MA'], label='Second Derivative 50-day MA', linestyle='--', color='pink')

plt.xlabel('Date')
plt.ylabel('Change')
plt.title('Derivatives and Moving Averages')
plt.legend()
plt.grid(True)
plt.show()
