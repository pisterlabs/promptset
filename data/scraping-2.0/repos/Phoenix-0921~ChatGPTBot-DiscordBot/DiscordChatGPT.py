import os
from dotenv import load_dotenv
import discord
from discord.ext import commands
import openai

load_dotenv()

# 需要創立.env 將DiscordToken 和 OPENAI_API 填入
token = os.environ.get('DiscordToken')
openai.api_key = os.environ.get('OPENAI_API')

# intents是要求機器人的權限
intents = discord.Intents.all()
# 建立 Discord 機器人連接,給權限
bot = commands.Bot(command_prefix='/',intents = intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    
# /ask 是前綴字 需輸入才能做詢問
@bot.command()
async def ask(ctx, *, question):
    if question:
        # 使用 ChatGPT 
        response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # 使用 ChatGPT 引擎
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
                ],
        max_tokens=150  # 根據需要調整
        )

        answer = response.choices[0].message.content

        # 將回答發送回去Discord
        await ctx.send(answer)
    else:
        await ctx.send("請再次詢問")

bot.run(token)
