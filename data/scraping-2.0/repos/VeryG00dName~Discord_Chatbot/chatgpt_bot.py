from dotenv import load_dotenv
import openai
import discord
import os
from discord.ext import commands

load_dotenv()

intents = discord.Intents.default()
intents.members = True
intents.message_content = True
openai.api_key = os.environ.get("openai-api")
discord_token = os.environ.get("discord-token")
global pre_prompt
pre_prompt = ""

class ChatGPTBot(commands.Bot):
    async def on_message(self, message):
        global pre_prompt
        # Handle incoming messages here
        print(message.author,message.channel,message.content)
        if message.author == self.user:
            return
        
        if message.content.startswith("!bot"):
            # This is a message intended for the bot, so we can respond
            # Retrieve the previous messages in the current channel
            channel = message.channel
            message_history = []
            async for previous_message in channel.history(limit=20):
                # Add the previses messeges to the message history
                # print(message_history)
                if previous_message.content == "context block":
                    break
                if previous_message.author != self.user:
                    dirty_user_input = previous_message.content.split()
                    clean_user_input = ' '.join(dirty_user_input[1:])
                    message_history.append({"role": "user", "content": clean_user_input})
                else:
                    message_history.append({"role": "assistant", "content": previous_message.content})
            message_history.reverse()
            if pre_prompt != "":
                message_history.append({"role": "system", "content": pre_prompt})
            # Use the message history as input for the chatbot
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message_history
             )

            # Check if the chatbot's response is not empty or whitespace-only
            if response.choices[0].message.content.strip():
                # Send the chatbot's response
                await message.channel.send(response.choices[0].message.content)
            else:
                # The chatbot's response is empty or whitespace-only, so we don't send it
                return
        elif message.content.startswith("!pre"):
            words = message.content.split()
            pre_prompt = ' '.join(words[1:])
            await bot.change_presence(activity=discord.Game(name=pre_prompt))
            return
        elif message.content.startswith("!art"):
       # This is a message requesting an image, so we generate one using the stable diffusion model

            # Generate the image
            words = message.content.split()
            new_prompt = ' '.join(words[1:])
            print(new_prompt)
            image = openai.Image.create(
	  		    prompt=new_prompt,
	            n=2,
	            size="1024x1024"
		    )

            # Send the image to the user
            await message.channel.send(image["data"][0]["url"])
        else:
            # This is not a message intended for the bot, so we ignore it
            # await message.channel.send("please start your message with !bot or !art Thank you.")
            return

bot = ChatGPTBot("!",intents=intents)
bot.run(discord_token)
