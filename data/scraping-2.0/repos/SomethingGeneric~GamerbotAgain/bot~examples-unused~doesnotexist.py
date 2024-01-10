# Pip
import disnake
from disnake.ext import commands, tasks
import openai
import requests
from random import randint

# Mine
from .util_functions import *


class DoesNotExist(commands.Cog):
    """Haha image generation"""

    def __init__(self, bot):
        self.bot = bot

    @commands.slash_command()
    async def dalle(self, inter, *, prompt: str):
        """Image generator thingie"""
        try:
            await inter.response.defer()
            openai.api_key = config["openai_key"]
            response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
            image_url = response["data"][0]["url"]
            r = requests.get(image_url)

            fn = f"dalle-out-{str(randint(1,1000))}.png"

            open(fn, "wb").write(r.content)

            await inter.send(f"You said: `{prompt}`.", file=disnake.File(fn))
            os.remove(fn)
        except Exception as e:
            await inter.send(f"Error: ```{str(e)}```")


def setup(bot):
    print("Loading DNE ext")
    bot.add_cog(DoesNotExist(bot))
