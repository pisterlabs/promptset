import os

import discord
import dotenv
from discord.ext import commands

import openai


class ImageGeneratorCog(commands.Cog):
    def __init__(self, bot : commands.Bot) -> None:
        self.bot = bot 
        
        
    @commands.hybrid_command(name="generate")
    async def generate_image_command(self, ctx : commands.Context, *, prompt : str) -> discord.Message:
        await ctx.defer()
        
        response = openai.Image.create(
            prompt=prompt,
            size="512x512"
        )
        
        image = response['data'][0]['url']
        
        return await ctx.send(image)
    


class Bot(commands.Bot):
    def __init__(self) -> None:
        super().__init__(command_prefix='!', intents=discord.Intents.all())


    async def setup_hook(self) -> None:
        await self.add_cog(ImageGeneratorCog(self))
        await self.tree.sync()

    async def on_ready(self) -> None:
        print('Je suis en ligne !')



def main() -> None:
    dotenv.load_dotenv()
    bot = Bot()

    TOKEN = os.getenv('TOKEN')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    if OPENAI_API_KEY is None:
        raise ValueError('Le token OpenAI n\'est pas défini !')

    if TOKEN is None:
        raise ValueError('Le token n\'est pas défini !')

    openai.api_key = OPENAI_API_KEY

    bot.run(TOKEN)


if __name__ == '__main__':
    main()