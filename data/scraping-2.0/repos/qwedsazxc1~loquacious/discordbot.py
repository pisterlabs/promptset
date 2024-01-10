import discord
from dotenv import load_dotenv
import asyncio
from discord.ext import commands
import os
import openai

load_dotenv()  # 加载 .env 文件
bot_token = os.getenv('BOT_TOKEN')

intents = discord.Intents.all()
bot = commands.Bot(command_prefix = "%", intents = intents)

# 當機器人完成啟動時
@bot.event
async def on_ready():
    print(f"目前登入身份 --> {bot.user}")

# 載入指令程式檔案
@bot.command()
async def load(ctx, extension):
    await bot.load_extension(f"cogs.{extension}")
    await ctx.send(f"Loaded {extension} done.")

# 卸載指令檔案
@bot.command()
async def unload(ctx, extension):
    await bot.unload_extension(f"cogs.{extension}")
    await ctx.send(f"UnLoaded {extension} done.")

# 重新載入程式檔案
@bot.command()
async def reload(ctx, extension):
    await bot.reload_extension(f"cogs.{extension}")
    await ctx.send(f"ReLoaded {extension} done.")

@bot.command()
async def hello(ctx):
    await ctx.send("hello world.")

# 一開始bot開機需載入全部程式檔案
async def load_extensions():
    for filename in os.listdir("./cogs"):
        if filename.endswith(".py"):
            await bot.load_extension(f"cogs.{filename[:-3]}")
            openai.api_key = "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"

async def main():
    async with bot:
        await load_extensions()
        await bot.start(bot_token)


if __name__ == "__main__":
    asyncio.run(main())
