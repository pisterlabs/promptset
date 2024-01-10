import discord
import openai

openai.api_key = ""  # Input OpenAI Key here in quotes as a string
discord_token = "DISCORD TOKEN HERE"  # Input Discord Token here
model_name = "ENGINE MODEL NAME HERE"  # Input Engine Model Name here 

intents = discord.Intents().all()
client = discord.Client(intents=intents)

responses = {}


@client.event
async def on_ready():
    print(f'SUCCESSFULLY logged in as {client.user}')

    
@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content == 'bots.shutdown': 
        await message.channel.send('Shutting down...')
        await client.close()
    if message.content.startswith('!'):  # Start discord message with ! to prompt chatbot
        prompt = message.content[1:]
        response = openai.Completion.create(  # See API documentation for further parameters
            engine=model_name,  
            prompt=prompt + ' ->',
            max_tokens=100,
            n=1,
            temperature=0.8,
            stop=["\n"]
        )
        print(response["choices"][0]["text"])
        message_to_send = response["choices"][0]["text"]
        await message.channel.send(message_to_send)


@client.event
async def on_reaction_add(reaction, user):
    if reaction.message.author == client.user:
        if str(reaction.emoji) == "👍" or str(reaction.emoji) == "👎":
            
            prompt = reaction.message.content
            response = openai.Completion.create(
                engine=model_name,
                prompt=prompt,
                max_tokens=100,
                n=1,
                temperature=0.8,
                stop="\n",
                logprobs=10 
            )
            if len(response.choices) > 0:
                logprobs = response.choices[0].logprobs.token_logprobs
                reward = 1 if str(reaction.emoji) == "👍" else -1
                for i, token_logprobs in enumerate(logprobs):
                    token = response.choices[0].text[i]
                    if isinstance(token_logprobs, dict):
                        token_reward = token_logprobs[token]["token_logprob"] * reward
                        openai.Completion.create(
                            engine=model_name, 
                            prompt=prompt + token,
                            max_tokens=0,
                            n=1,
                            logprobs=10,
                            echo=True,
                            stop="\n",
                            temperature=0,
                            stop_sequence="\n",
                            presence_penalty=0.0,
                            frequency_penalty=0.0,
                            stop_penalty=0.0,
                            logit_bias={token: token_reward}
                        )
                if reward == 1:
                    await reaction.message.channel.send(f'{user} reinforced the response: "{prompt}"')
                else:
                    await reaction.message.channel.send(f'{user} penalized the response: "{prompt}"')


client.run(discord_token)
