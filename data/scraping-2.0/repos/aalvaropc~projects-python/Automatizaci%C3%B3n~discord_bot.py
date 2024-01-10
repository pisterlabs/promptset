import discord
import openai

intents = discord.Intents.default()
intents.message_content = True

token = '_tu_token_'
openai.api_key = 'tu_token_api'

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print("Logeado como {0.user}".format(client))

@client.event
async def on_message(message):
    username = str(message.author).split('#')[0]
    user_message = str(message.content)
    channel = str(message.channel.name)

    print(username + " Alvaro " + user_message.lower() + " in " + channel)

    if message.channel.name == "bot":
        response = openai.Completion.create(
            model = "text-davinci-003",
            prompt = user_message,
            max_tokens = 3000,
            temperature = 0.7
        )
        output = response["choices"][0]["text"]
        
        print(output)
        
        await message.channel.send(output)
    
client.run(token)