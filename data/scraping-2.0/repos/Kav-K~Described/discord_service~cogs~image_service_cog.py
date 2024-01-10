import pickle
import re
import traceback
from collections import defaultdict
from pathlib import Path

import aiofiles
import discord

from discord_service.embeds.embed_helper import EmbedStatics
from services.check_service import Check
from services.environment_service import EnvService
from services.openai_service import OpenAIExecutor


class ServerInformation:
    def __init__(self, status: bool = False):
        self.status = status


class ImageService(discord.Cog, name="ImageService"):
    """cog containing the optimizer command"""

    async def change_guild_status(self, guild_id, status: bool):
        self.server_information[guild_id].status = status

        try:
            directory_path = Path(EnvService.save_path()) / "pickles"
            directory_path.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(
                EnvService.save_path() / "pickles" / "server_information.pickle",
                "wb",
            ) as f:
                await f.write(pickle.dumps(self.server_information))
                return True
        except:
            traceback.print_exc()
            print("Could not save server information to disk after update.")
            return False

    def __init__(
        self,
        bot,
    ):
        super().__init__()
        self.bot = bot
        self.openai_service = OpenAIExecutor()
        self.allowed_channels = EnvService.get_described_channels()

        try:
            with open(
                EnvService.save_path() / "pickles" / "server_information.pickle",
                "rb",
            ) as f:
                self.server_information = pickle.load(f)
                print("Loaded server information pickle.")
        except:
            self.server_information = defaultdict(ServerInformation)
            for guild in self.bot.guilds:
                self.server_information[guild.id] = ServerInformation(False)

    @discord.slash_command(
        name="describe",
        description="Turn image descriptions on or off for the server.",
        guild_ids=EnvService.get_allowed_guilds(),
        checks=[Check.check_admin_roles()],
    )
    @discord.option(
        name="status",
        description="Turn descriptions on or off",
        required=False,
        choices=["on", "off"],
    )
    @discord.guild_only()
    async def described_command(self, ctx: discord.ApplicationContext, status: str):
        """Command handler. Given a string it generates an output that's fitting for image generation"""
        if status == "on" or status == "off":
            if await self.change_guild_status(
                ctx.guild_id, True if status == "on" else False
            ):
                await ctx.respond(
                    embed=EmbedStatics.build_status_change_success_embed(
                        self.server_information[ctx.guild_id].status
                    )
                )
                return
        else:
            await ctx.respond(
                embed=EmbedStatics.build_status_display_embed(
                    self.server_information[ctx.guild_id].status
                )
            )
            return

        await ctx.respond(
            embed=EmbedStatics.build_status_set_failure_embed(
                "There was an error changing the status of image descriptions for this server."
            )
        )

    @discord.Cog.listener()
    async def on_message(self, message: discord.Message):
        # Check if the message is from a bot.
        if message.author == self.bot.user:
            return

        if not message.guild:
            return

        if message.type != discord.MessageType.default:
            return

        # Check if the message is from a guild.
        if not message.guild:
            return

        if message.content.strip().startswith("~"):
            return

        if not self.server_information[message.guild.id].status:
            return

        if not message.channel.name in self.allowed_channels:
            return

        image_urls = []

        if len(message.attachments) > 0:
            for _file in message.attachments:
                _file: discord.Attachment
                if _file.content_type.startswith("image"):
                    image_urls.append(_file.url)

        if len(image_urls) > 0:
            print(
                "Sending an image description request for message URL: "
                + message.jump_url
            )

            # Add a reaction to the message to denote processing
            try:
                await message.add_reaction("🔃")
            except:
                pass

            # Send an image description request
            for url in image_urls:
                print("Processing " + str(url))
                try:
                    response = await self.openai_service.send_image_evaluation_request(
                        [url]
                    )
                    await message.reply(
                        embed=EmbedStatics.build_described_image_embed(
                            message, url, response
                        )
                    )
                except Exception as e:
                    traceback.print_exc()
                    await message.reply(
                        embed=EmbedStatics.build_image_analysis_failure_embed(str(e))
                    )

            try:
                await message.delete()
            except:
                pass
