import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import discord
from discord.commands import slash_command
from discord.ext import commands

from marsbots import config
from marsbots.discord_utils import get_discord_messages
from marsbots.discord_utils import in_channels
from marsbots.discord_utils import update_message
from marsbots.discord_utils import wait_for_user_reply
from marsbots.language_models import complete_text
from marsbots.language_models import OpenAIGPT3LanguageModel
from marsbots.models import ChatMessage
from marsbots.settings_manager import LocalSettingsManager


class ButtonView(discord.ui.View):
    def __init__(self):
        # making None is important if you want the button work after restart!
        super().__init__(timeout=None)

    # custom_id is required and should be unique for <commands.Bot.add_view>
    # attribute emoji can be used to include emojis which can be default str emoji or str(<:emojiName:int(ID)>)
    # timeout can be used if there is a timeout on the button interaction. Default timeout is set to 180.
    @discord.ui.button(
        style=discord.ButtonStyle.blurple,
        custom_id="counter:firstButton",
        label="Button",
    )
    async def leftButton(self, button, interaction):
        await interaction.response.edit_message(content="button was pressed!")


@dataclass
class ExampleBotSettings:
    setting1: int = 10
    setting2: int = 10


class ExampleCog(commands.Cog):
    def __init__(self, bot: commands.bot) -> None:
        self.bot = bot
        self.language_model = OpenAIGPT3LanguageModel(config.LM_OPENAI_API_KEY)
        self.settings_path = Path("./examplebot_settings.json")
        self.settings_manager = LocalSettingsManager(
            self.settings_path,
            defaults=ExampleBotSettings(),
        )

    @commands.command()
    async def get_commands(self, ctx) -> None:
        print([c.qualified_name for c in self.walk_commands()])

    @commands.command()
    async def whereami(self, ctx) -> None:
        await ctx.send("Hello from a custom cog")
        await ctx.send(ctx.guild.id)

    @slash_command(guild_ids=[config.TEST_GUILD_ID])
    async def howami(self, ctx) -> None:
        await ctx.respond("doing great")

    @commands.command()
    async def get_messages(self, ctx: commands.Context) -> None:
        messages = await get_discord_messages(ctx.channel, 10)
        for message in messages:
            msg = ChatMessage(
                content=message.content,
                sender=message.author.name,
            )
            print(msg)

    @commands.command()
    async def complete(
        self,
        ctx: commands.context,
        max_tokens: int,
        *input_text: str,
    ) -> None:
        prompt = " ".join(input_text)
        async with ctx.channel.typing():
            completion = complete_text(self.language_model, prompt, max_tokens)
            await ctx.send(prompt + completion)

    @slash_command(guild_ids=[config.TEST_GUILD_ID])
    async def complete_some_text(
        self,
        ctx,
        max_tokens: int,
        prompt: str,
    ) -> None:
        completion = await complete_text(self.language_model, prompt, max_tokens)
        print(prompt + completion)
        await ctx.respond(prompt + completion)

    @slash_command(
        guild_ids=[config.TEST_GUILD_ID],
        name="slash_command_name",
        description="command description!",
    )
    async def button(self, ctx):
        navigator = ButtonView()
        await ctx.respond("press the button.", view=navigator)

    @commands.command()
    async def resolve(self, ctx, message_id):
        msg = await ctx.fetch_message(message_id)
        print(msg.content)

    @commands.command()
    @in_channels([config.TEST_CHANNEL_ID])
    async def test_in_channels(self, ctx):
        await ctx.send("In the test channel.")

    @commands.command()
    async def test_edit_message(self, ctx):
        assets_path = os.path.join(os.path.dirname(__file__), "assets")
        filepaths = [
            os.path.join(assets_path, fname)
            for fname in ["be-patient.png", "s-l1600.jpg", "uc2.png", "s-l1600 (1).jpg"]
        ]
        files = [discord.File(filepath) for filepath in filepaths[:2]]
        message = await ctx.send("Hey", files=files)
        await asyncio.sleep(3)
        await update_message(message, content="Goodbye", image_paths=filepaths[2:])

    @commands.command()
    async def get_setting_1(self, ctx):
        setting = self.settings_manager.get_setting(ctx.guild.id, "setting1")
        await ctx.send(setting)

    @commands.command()
    async def get_channel_setting_1(self, ctx):
        setting = self.settings_manager.get_channel_setting(
            ctx.channel.id,
            ctx.guild.id,
            "setting1",
        )
        await ctx.send(setting)

    @commands.command()
    async def get_settings(self, ctx):
        print(self.settings_manager.settings)

    @commands.command()
    async def update_setting_1(self, ctx, value):
        self.settings_manager.update_setting(ctx.guild.id, "setting1", value)
        await ctx.send("updated setting1")

    @commands.command()
    async def update_channel_setting_1(self, ctx, value):
        self.settings_manager.update_channel_setting(
            ctx.channel.id,
            ctx.guild.id,
            "setting1",
            value,
        )
        await ctx.send("updated channel setting1")

    @slash_command(guild_ids=[config.TEST_GUILD_ID])
    async def update_settings(
        self,
        ctx,
        setting: discord.Option(
            str,
            description="Setting name to update",
            required=True,
            choices=list(ExampleBotSettings.__dataclass_fields__.keys()),
        ),
        channel_name: discord.Option(
            str,
            description="Channel to update setting for",
            required=False,
        ),
    ):

        if channel_name:
            await self.handle_update_channel_settings(ctx, setting, channel_name)
        else:
            await self.handle_update_settings(ctx, setting)

    async def handle_update_settings(self, ctx, setting):
        await ctx.respond(
            f"Enter a new value for {setting}. (Currently"
            f" {self.settings_manager.get_setting(ctx.guild.id, setting)})",
        )
        resp = await wait_for_user_reply(self.bot, ctx.author.id)
        try:
            new_val = ExampleBotSettings.__dataclass_fields__[setting].type(
                resp.content,
            )
        except ValueError:
            await ctx.send(f"{resp.content} is not a valid value for {setting}")
            return
        self.settings_manager.update_setting(ctx.guild.id, setting, new_val)
        await ctx.send(f"Updated {setting} to {new_val}")

    async def handle_update_channel_settings(self, ctx, setting, channel_name):
        channel = discord.utils.get(ctx.guild.channels, name=channel_name)
        if not channel:
            await ctx.respond(f"No channel named {channel_name}")
            return

        await ctx.respond(
            f"Enter a new value for {setting}. (Currently"
            f" {self.settings_manager.get_channel_setting(channel.id, ctx.guild.id, setting)})",
        )
        resp = await wait_for_user_reply(self.bot, ctx.author.id)
        try:
            new_val = ExampleBotSettings.__dataclass_fields__[setting].type(
                resp.content,
            )
        except ValueError:
            await ctx.send(f"{resp.content} is not a valid value for {setting}")
            return
        self.settings_manager.update_channel_setting(
            channel.id,
            ctx.guild.id,
            setting,
            new_val,
        )
        await ctx.send(f"Updated {setting} to {new_val}")


def setup(bot: commands.Bot) -> None:
    bot.add_cog(ExampleCog(bot))
