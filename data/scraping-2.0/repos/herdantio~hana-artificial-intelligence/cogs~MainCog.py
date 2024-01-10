import openai
from discord.ext import commands

from utils.EnvironmentLoader import load_env

env = load_env()

DISCORD_GENERAL_CHANNEL_ID = env["DISCORD_GENERAL_CHANNEL_ID"]


class MainCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_ready(self):
        await self.bot.get_channel(DISCORD_GENERAL_CHANNEL_ID).send(
            f"{self.bot.user} has connected to Discord!"
        )

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user:
            return

        prompt = [
            {
                "role": "system",
                "content": "you are my female secretary who's helping me with my work. you are a gentle person who likes to help others.",
            },
            {
                "role": "user",
                "content": message.content,
            },
        ]
        async with message.channel.typing():
            res = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=prompt)
        if res is not None:
            await message.channel.send(
                "\n".join([choice.message.content for choice in res.choices])
            )
        else:
            await message.channel.send("Sorry, there was an error.")


async def setup(bot):
    await bot.add_cog(MainCog(bot))
