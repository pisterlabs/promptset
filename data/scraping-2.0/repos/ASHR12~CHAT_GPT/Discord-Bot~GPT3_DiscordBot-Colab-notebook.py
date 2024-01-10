!pip install discord.py
!pip install openai
!pip install nest_asyncio

import discord
import openai
import nest_asyncio

# Apply the nest_asyncio library to allow nested event loops
nest_asyncio.apply()

# Set up your Discord bot token and OpenAI API key
DISCORD_BOT_TOKEN = 'provide discord bot token'
OPENAI_API_KEY = 'provide open ai token'

# Create an instance of the Intents class and specify which intents to enable
intents = discord.Intents.all()

# Initialize the Discord client with the specified intents and OpenAI API
client = discord.Client(intents=intents)
openai.api_key = OPENAI_API_KEY

@client.event
async def on_message(message):
    # Ignore messages sent by the bot itself
    if message.author == client.user:
        return

    # Check if the message starts with `gpt3:`
    if message.content.startswith('gpt3:'):
        # Get the prompt from the message (remove `gpt3:` from the start)
        prompt = message.content[5:]

        # Use the OpenAI API to generate a response
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.5,
            )

            # Send the generated response back in the Discord channel
            await message.channel.send(response['choices'][0]['text'])
        except Exception as e:
            await message.channel.send(f"An error occurred: {e}")

# Start the Discord bot
client.run(DISCORD_BOT_TOKEN)