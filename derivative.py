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

# Plot the first and second derivatives over time
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['First_Derivative'], label='First Derivative', color='blue')
plt.plot(df.index, df['Second_Derivative'], label='Second Derivative', color='green')
plt.xlabel('Date')
plt.ylabel('Change')
plt.title('First and Second Derivatives of "Adj Close"')
plt.legend()
plt.grid(True)
plt.show()
