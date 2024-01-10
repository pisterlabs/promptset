import discord
from OpenaiAgent.OpenaiClient import OpenaiClient, LangchainClient

from .DiscordHelper import DiscordHelper, DiscordClientConfig

# Define other agents
# openai_client = OpenaiClient("/mnt/d/Dev/discord-ai-bot/OpenaiClientConfig.json")
langchain_client = LangchainClient("/mnt/d/Dev/discord-ai-bot/OpenaiClientConfig.json")

# Define environment

MESSAGE_CHANNEL_MONITORED = ["bot"]


# Define your desired intents. For example, to receive message events:
intents = discord.Intents.default()
intents.message_content = True

# Create a Discord client with the specified intents
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    # ignore messages from self
    if message.author == client.user:
        return
    # ignore messages from other channels
    if message.channel.name not in MESSAGE_CHANNEL_MONITORED:
        return
    if message.content.startswith('!hello'):
        await message.channel.send('Hello!')
        return
    if message.content == "!reset" :
        langchain_client.clear_memory()
        await message.channel.send("Memory cleared!")
        return
    print(f"Received message: {message.content}")
    response = langchain_client.chat(message.content)
    await DiscordHelper.send_over_length_message(message.channel, response)
    print(f"Sent response: {response}")

# Run the bot with your token
client_config = DiscordClientConfig("/mnt/d/Dev/discord-ai-bot/DiscordClientConfig.json")
client.run(client_config.get_config()["token"])