# %%
import datetime as dt
import talib as ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# %%
ticker = "MSFT"
end_date = dt.datetime.now() - dt.timedelta(days=365*2)
start_date = end_date - dt.timedelta(days=365*5)
df = pd.DataFrame(yf.download(ticker, start_date, end_date))
df.head(10)

# %%
df['RSI'] = ta.RSI(df['Close'])
df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = ta.MACD(df['Close'])

# %%
df.shape

# %%
df.isnull().sum()

# %%
df = df.dropna()

# %%
df.isnull().sum()

# %%
df.shape

# %%
df.columns

# %%
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import backtesting
backtesting.set_bokeh_output(notebook=False)
from backtesting import set_bokeh_output
set_bokeh_output(notebook=False)

# %%
class QLearningStrategy(Strategy):
    upper_bound = 70
    lower_bound = 30
    
    def init(self):
        price = self.data.Close
        self.macd = self.I(lambda x: ta.MACD(x)[0], price)
        self.macd_signal = self.I(lambda x: ta.MACD(x)[1], price)
        self.rsi = self.I(lambda x: ta.RSI(x), price)

    def next(self):
        if crossover(self.macd, self.macd_signal) and crossover(self.rsi, self.upper_bound):
            self.buy()
        elif crossover(self.macd_signal, self.macd) and crossover(self.lower_bound, self.rsi):
            self.sell()

# %%
money = 10000

# %%
bt = Backtest(df, QLearningStrategy, cash=money, commission=.002, exclusive_orders=True)
stats = bt.run()
print(stats)

# %%
stats[3]

# %%
stats["Sharpe Ratio"]

# %%
type(stats["Equity Final [$]"])

# %%
print(f'Reward:{(stats["Equity Final [$]"] - money)/money}')

# %%
file_path = "denek.txt"
with open(file_path, "w") as file:
    file.write(str(stats))
print(f"Statistics have been saved to {file_path}")

# %%
bt.plot()


