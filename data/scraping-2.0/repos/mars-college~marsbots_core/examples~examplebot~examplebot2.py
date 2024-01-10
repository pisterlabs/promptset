from discord.ext import commands

from marsbots import config
from marsbots.language_models import OpenAIGPT3LanguageModel


class ExampleCog2(commands.Cog):
    def __init__(self, bot: commands.bot) -> None:
        self.bot = bot
        self.language_model = OpenAIGPT3LanguageModel(config.LM_OPENAI_API_KEY)

    @commands.command()
    async def example(self, ctx: commands.context) -> None:
        await ctx.send("Hello world from cog 2!")


def setup(bot: commands.Bot) -> None:
    bot.add_cog(ExampleCog2(bot))
