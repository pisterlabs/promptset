import discord
from discord.ext import commands
import openai
from utils.constants import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


class JokeCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.prompt = """
The following is a list of very funny jokes:
Which bear is the most condescending? A pan-duh!
What kind of noise does a witch’s vehicle make? Brrrroooom, brrroooom.
What’s brown and sticky? A stick.
Two guys walked into a bar. The third guy ducked.
How do you get a country girl’s attention? A tractor.
Why are elevator jokes so classic and good? They work on many levels.
What do you call a pudgy psychic? A four-chin teller.
What did the police officer say to his belly-button? You’re under a vest.
What do you call it when a group of apes starts a company? Monkey business.
"""

    @commands.command()
    @commands.cooldown(1, 30, commands.BucketType.user)
    async def dadjoke(self, ctx):
        if not ctx.channel.nsfw:
            await ctx.send("This command can only be used in NSFW channels!")
            return

        prompt = self.prompt

        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            temperature=0.8,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0,
            stop=['\n']
        )

        output = response['choices'][0]['text']

        await ctx.send(output)


def setup(bot):
    bot.add_cog(JokeCog(bot))
