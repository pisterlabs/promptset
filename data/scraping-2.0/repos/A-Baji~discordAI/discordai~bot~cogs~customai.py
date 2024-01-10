""""
Copyright © Krypton 2019-2022 - https://github.com/kkrypt0nn (https://krypton.ninja)
Description:
🐍 A simple template to start to code your own and personalized discord bot in Python programming language.

Version: 5.4.1
"""

from discord import app_commands
from discord.ext import commands
from discord.ext.commands import Context

import openai


class CustomAI(commands.Cog, name="customai"):
    def __init__(self, bot):
        self.bot = bot

    @commands.hybrid_command(
        name="customai",
        description="Generate a custom openAI completion",
    )
    @app_commands.describe(
        model="The id of your custom model",
        prompt="The prompt to pass to your model: Default=\"\"",
        temp="What sampling temperature to use. Higher values means more risks: Min=0 Max=1 Default=1",
        presence_penalty="Number between -2.0 and 2.0. Positive values will encourage new topics: Min=-2 Max=2 Default=0",
        frequency_penalty="Number between -2.0 and 2.0. Positive values will encourage new words: Min=-2 Max=2 Default=0",
        max_tokens="The max number of tokens to generate. Each token costs credits: Default=125",
        stop="Whether to stop after the first sentence: Default=false",
        openai_key="The openai key associated with the given model: Default=config.openai_key")
    async def customai(self, context: Context, model: str = "", prompt: str = "", temp: float = 1.0,
                       presence_penalty: float = 0.0, frequency_penalty: float = 0.0, max_tokens: int = 125,
                       stop: bool = False, openai_key: str = ""):
        temp = min(max(temp, 0), 1)
        presPen = min(max(presence_penalty, -2), 2)
        freqPen = min(max(frequency_penalty, -2), 2)

        await context.defer()
        try:
            openai.api_key = openai_key or self.bot.config["openai_key"]
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                temperature=temp,
                frequency_penalty=presPen,
                presence_penalty=freqPen,
                max_tokens=max_tokens,
                echo=True if prompt else False,
                stop='.' if stop else None,
            )
            await context.send(f"Model: {model}\n{'Prompt: ' + prompt if prompt else ''}\n{response['choices'][0]['text'][:2000]}")
        except Exception as error:
            print(f"Failed to generate valid response for prompt: {prompt} with model: {model}\nError: {error}")
            await context.send(
                f"Failed to generate valid response for prompt: {prompt} with model: {model}\nError: {error}"
            )


async def setup(bot):
    await bot.add_cog(CustomAI(bot))
