import datetime as dt
import talib as ta
import pandas as pd
import yfinance as yf
import numpy as np
import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.preprocessing import StandardScaler  # For normalization
from stable_baselines3.dqn import MlpPolicy  # Import the DQN policy
import matplotlib.pyplot as plt

# Define trading parameters
ticker = "AAPL"
end_date = dt.datetime.now() - dt.timedelta(days=365 * 2)
start_date = end_date - dt.timedelta(days=365 * 5)
df = pd.DataFrame(yf.download(ticker, start_date, end_date))
df['SMA'] = ta.SMA(df['Close'])
df['EMA'] = ta.EMA(df['Close'])
df['RSI'] = ta.RSI(df['Close'])
df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = ta.MACD(df['Close'])
df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'])
df['OBV'] = ta.OBV(df['Close'], df['Volume'])
df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'])
df = df.dropna()

# Normalize the state space
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Define the RL environment
class TradingEnv(gym.Env):
    def __init__(self, df, commission=0.002, slippage=0.001):
        super(TradingEnv, self).__init__()

        self.df = df
        self.current_step = 0
        self.initial_balance = 10000  # Initial balance for the agent
        self.balance = self.initial_balance
        self.shares_held = 0
        self.max_shares = 1000  # Maximum number of shares the agent can hold
        self.commission = commission  # Transaction cost percentage
        self.slippage = slippage  # Slippage percentage
        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(len(df.columns),), dtype=np.float32
        )
        self.done = False  # Initialize the 'done' attribute

    def reset(self):
        # Reset the environment to its initial state
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        # Initialize and return the initial state
        initial_state = self.df.iloc[0].values
        return initial_state

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        if action == 0:  # Buy
            if (
                self.balance >= self.df.iloc[self.current_step]["Close"]
                and self.shares_held < self.max_shares
            ):
                # Calculate transaction cost
                transaction_cost = (
                    self.df.iloc[self.current_step]["Close"] * self.commission
                )
                # Calculate slippage
                slippage_cost = (
                    self.df.iloc[self.current_step]["Close"] * self.slippage
                )
                total_cost = transaction_cost + slippage_cost
                shares_bought = (
                    self.balance - total_cost
                ) // self.df.iloc[self.current_step]["Close"]
                if shares_bought > 0:
                    self.shares_held += shares_bought
                    self.balance -= (
                        shares_bought * self.df.iloc[self.current_step]["Close"]
                        + total_cost
                    )
        elif action == 1:  # Sell
            if self.shares_held > 0:
                # Calculate transaction cost
                transaction_cost = (
                    self.df.iloc[self.current_step]["Close"] * self.commission
                )
                # Calculate slippage
                slippage_cost = (
                    self.df.iloc[self.current_step]["Close"] * self.slippage
                )
                total_cost = transaction_cost + slippage_cost
                shares_sold = self.shares_held
                self.balance += (
                    shares_sold * self.df.iloc[self.current_step]["Close"]
                ) - total_cost
                self.shares_held = 0

        # Calculate profit or loss
        self.profit_or_loss = (
            self.balance + self.shares_held * self.df.iloc[self.current_step]["Close"]
        ) - self.initial_balance

        # Get the next state
        next_state = self.df.iloc[self.current_step].values

        # Calculate the reward as profit or loss
        reward = self.profit_or_loss

        return next_state, reward, self.done, {}

# Create the RL environment
# env = TradingEnv(df)

# Now, you can use this environment to train an RL agent using a suitable RL library like Stable Baselines, TensorFlow, or PyTorch.

# Training the RL agent is a separate and more extensive process, and it depends on the specific library you choose.
#  If you have a preference for an RL library or need further assistance with training, please let me know.

# Create the RL environment
env = TradingEnv(df)
env = DummyVecEnv([lambda: env])  # Wrap the environment

# Define the number of episodes for training
Time_Step = 1000000
episodes = 10
models_dir = 'D:/Code/Stock/Python/Stock/models/DQN'
logdir = 'D:/Code/Stock/Python/Stock/logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# # Define the DQN model
# model = DQN("MlpPolicy", env, verbose=1)  # You can customize the policy network architecture if needed

# Define the DQN model with customization
model = DQN(MlpPolicy, 
            env, 
            # learning_rate=0.001,        # Adjusted learning rate
            # exploration_fraction=0.1,   # Custom exploration fraction
            # exploration_final_eps=0.02, # Custom final exploration epsilon
            tensorboard_log=logdir,
            verbose=1)

# Train the DQN agent
# model.learn(total_timesteps=num_episodes)

model.learn(total_timesteps=Time_Step ,reset_num_timesteps=False, tb_log_name='DQN')

# Save the trained model
model.save("D:/Code/Stock/dqn_trained_model")

best_rew = -float('inf')

# Create lists to store episode rewards
episode_rewards = []

for i in range(episodes):
    obs = env.reset()
    done = False
    episode_rew = 0
    while not done:
        action, _ = model.predict(obs)
        obs, rew, done, _ = env.step(action)
        episode_rew += rew
    best_rew = max(best_rew, episode_rew)
    episode_rewards.append(episode_rew)

print(best_rew)
print(f'Best reward: {best_rew[0]:.2f}')

# Plot the improvement of the agent
plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes + 1), episode_rewards, marker='o', linestyle='-', color='b', label='Episode Reward')
plt.axhline(y=best_rew, marker='x', linestyle='--', color='r', label='Best Reward')
plt.title('DQN Agent Training Progress')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()  # Add a legend to distinguish between episode and best rewards
plt.grid(True)
plt.show()

# Now you have a trained DQN agent that can be used for making trading decisions.
#  You can also fine-tune the hyperparameters and policy network architecture as needed for better performance.

# # Train the DQN agent and collect episode rewards
# for episode in range(episodes):
#     obs = env.reset()
#     total_reward = 0.0
#     done = False
#     while not done:
#         action, _ = model.predict(obs)
#         obs, reward, done, _ = env.step(action)
#         total_reward += reward
#     episode_rewards.append(total_reward)