import discord
from discord.ext import commands
from discord import app_commands
from Database.GuildObjects import MikoMember
from OpenAI.ai import MikoGPT
from tunables import *
import os
from dotenv import load_dotenv
load_dotenv()


class ImageGeneratorCog(commands.Cog):
    def __init__(self, client):
        self.client: discord.Client = client

    @commands.Cog.listener()
    async def on_ready(self):
        self.tree = app_commands.CommandTree(self.client)


    @app_commands.command(name="image", description=f"{os.getenv('APP_CMD_PREFIX')}Generate an image based off a prompt")
    @app_commands.guild_only
    @app_commands.describe(
        prompt='Prompt to generate image from'
    )
    async def generate_image(self, interaction: discord.Interaction, prompt: str):
        
        u = MikoMember(user=interaction.user, client=interaction.client, check_exists=False)
        gpt = MikoGPT(u=u, client=interaction.client, prompt=f"i:{prompt}")
        await gpt.respond(interaction=interaction)
        


    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        u = MikoMember(user=interaction.user, client=interaction.client)
        await u.ainit()
        if (await u.profile).cmd_enabled('IMAGE_GENERATION') != 1:
            await interaction.response.send_message(content=tunables('GENERIC_BOT_DISABLED_MESSAGE'), ephemeral=True)
            return False

        await interaction.response.send_message(content=tunables('LOADING_EMOJI'), silent=True)
        return True


async def setup(client: commands.Bot):
    await client.add_cog(ImageGeneratorCog(client))