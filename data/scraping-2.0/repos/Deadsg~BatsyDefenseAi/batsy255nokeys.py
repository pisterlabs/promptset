import os
import discord
import openai
import numpy as np
from sklearn.linear_model import LinearRegression
import gym

# Initialize Discord client and OpenAI API key
client = discord.Client()
openai.api_key = ""  # Replace with your actual API key

# Create a CartPole environment
env = gym.make('CartPole-v1')

# Event handler for when the bot is ready
@client.event
async def on_ready():
    print(f'Bot is ready. Logged in as {client.user}')

# Event handler for when a message is received
@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("!llm"):
        # Extract the query from the message content
        query = message.content[5:].strip()

        if query == "cartpole":
            total_reward = 0
            done = False
            state = env.reset()

            while not done:
                action = env.action_space.sample()  # Random action for demonstration
                state, reward, done, _ = env.step(action)
                total_reward += reward

            result_text = f'Total Reward: {total_reward}'

            # Send the result back to the Discord channel
            await message.channel.send(result_text)

        elif query == "linear_regression":
            # Generate some random data for demonstration
            np.random.seed(0)
            X = np.random.rand(100, 1)
            y = 2 * X.squeeze() + np.random.randn(100)

            # Create and fit the linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Get the coefficients
            slope = model.coef_[0]
            intercept = model.intercept_

            result_text = f'Linear Regression Model: y = {slope:.2f}x + {intercept:.2f}'

            # Send the result back to the Discord channel
            await message.channel.send(result_text)

# Run the bot with your Discord token
client.run("")  # Replace with your actual Discord token