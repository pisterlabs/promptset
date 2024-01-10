import os
import discord
import openai
from discord.ext import commands

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
prefix = "!"
bot = commands.Bot(command_prefix=prefix, intents=intents)

openai.api_key = os.environ.get("key_chatgpt")

# OpenAI
class ChatBotCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @bot.command()
    async def ask(self, ctx, *, question):
        try:
            conversation = [
                {"role": "user", "content": question},
                {"role": "system", "content": "MEU NOME E Role Aleatorio, E AGORA TAMBÉM SOU UM CHAT-BOT"}
            ]   

            response = openai.Completion.create(
                model="gpt-3.5-turbo",
                messages=conversation,
                max_tokens=1024
            )

            await ctx.send(response.choices[0].message.content)
        except Exception as error:
            await ctx.send(f"Ocorreu um erro: {str(error)}")
  
async def setup(bot):
    await bot.add_cog(ChatBotCog(bot))  