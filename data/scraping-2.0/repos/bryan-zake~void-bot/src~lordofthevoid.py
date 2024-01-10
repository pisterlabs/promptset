import discord
import time
import random
import openai
import os

openai.organization = os.environ.get("OPENAI_ORGANIZATION_ID")
openai.api_key = os.environ.get("OPENAI_API_KEY")
discord_token = os.environ.get("DISCORD_TOKEN")

def random_line(afile):
    line = next(afile)
    for num, aline in enumerate(afile, 2):
        if random.randrange(num):
            continue
        line = aline
    return line

def bot_response():
	resp = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		messages=[
			{"role": "system", "content": "You are the lord of the void."},
			{"role": "user", "content": "Write a one line poem about deleting a message"},
		],
        temperature=0.7
	)
	return resp['choices'][0]['message']['content']


class DiscordVoidBot(discord.Client):
    async def on_ready(self):
        print('Logged on as', self.user)

    async def on_message(self, message):
        if message.author == self.user:
            return
        else:
            wait_time = random.randint(0, 30)*.1
            t1 = time.time()
            try:   
                response = bot_response()
            except:
                # read random line from pre generated responses
                with open("pregenerated_responses.txt") as f:
                    response = random_line(f) 
            t2 = time.time()
            delta_t = t2 - t1
            if delta_t < wait_time:
                diffwait = wait_time - delta_t
                time.sleep(wait_time - delta_t)  
            channel = message.channel
            if channel.name == 'the-void':
                await message.delete()
                await channel.send(response)

intents = discord.Intents.default()
intents.message_content = True

voidbot = DiscordVoidBot(intents=intents)
voidbot.run(discord_token)
