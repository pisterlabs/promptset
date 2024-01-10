import discord
import openai
import logging

intents = discord.Intents.default()
intents.messages = True  # Enable the messages intent
client = discord.Client(intents=intents)

openai_api_key = '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'  # Make sure to put your actual OpenAI API key here
openai.api_key = openai_api_key

# Setup logging
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')


@client.event
async def on_message(message):
    print(f"Received a message: {message.content}")  # Logs every message the bot receives
    print(f"Received a message from {message.author}: {message.content}")
    if message.author == client.user:
        return

    if message.content.startswith('!askDitka'):
        try:
            user_question = message.content[len('!askDitka'):].strip()  # This grabs the text after the command
            print(f"Received question: {user_question}")  # Log the received question

            prompt = f"Imagine you are an angry Coach Mike Ditka responding to the following question: {user_question}\n\nResponse:"
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens=100
            )
            ditka_response = response.choices[0].text
            await message.channel.send(f"Coach Ditka says: {ditka_response}")
        except Exception as e:
            print(f"An error occurred: {e}")
            await message.channel.send("An error occurred while processing your request.")

client.run('MTEzNzE5OTg4OTk5OTIwODQ0OA.GHxCmt.bd_Mjd3KoN3r5cPQbx2XzzLgK_0HzrVBf39yEI')  # Make sure to put your actual Discord bot token here


print(f"Channel: {message.channel}")
print(f"Guild: {message.guild}")
