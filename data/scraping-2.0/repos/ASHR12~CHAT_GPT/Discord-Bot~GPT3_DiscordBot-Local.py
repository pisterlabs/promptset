import discord
import requests
import openai

intents = discord.Intents.all()

client = discord.Client(intents=intents)

openai.api_key = "provide your open ai token"

def get_gpt3_response(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return response['choices'][0]['text']
    except Exception as e:
        print(f'Error: {e}')
        return 'Sorry, something went wrong.'

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    if message.content.lower().startswith('gpt3'):
        prompt = message.content[5:]
        gpt3_response = get_gpt3_response(prompt)
        await message.channel.send(gpt3_response)

client.run('Provide your bot token')