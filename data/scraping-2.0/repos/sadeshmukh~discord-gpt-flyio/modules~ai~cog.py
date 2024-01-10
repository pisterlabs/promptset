import nextcord
from nextcord.ext import commands
from modules.storage.firebase import FirebaseStorage
import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

# utilities for gpt3 cog and other ai stuff - NO GROUPS


class AI(commands.Cog):
    def __init__(
        self,
        bot: commands.Bot,
        storage: FirebaseStorage,
    ):
        self.bot = bot
        self.guild_chat_history = storage.child("history").child("guild")
        self.dm_chat_history = storage.child("history").child("dm")
        self.default_system_message = storage.child("system").child("default")
        self.dm_system_messages = storage.child("system").child("dm")
        self.guild_system_messages = storage.child("system").child("guild")
        self.user_usage = storage.child("usage").child("user")
        self.global_usage = storage.child("usage").child("global")
        self.user_limit = storage.child("limits").child("user")
        self.global_limit = storage.child("limits").child("global")
        self.bot_channel = storage.child("bot_channel").child("guild")
        self.default_bot_channel = storage.child("bot_channel").child("default")

        self.ignored_users = storage.child("ignored_users")

        self.ai_params = storage.child("ai_params")
        self.openai_params = self.ai_params.child("openai")

    async def get_response(
        self,
        history=[],
        system=None,
    ):
        if not system:
            system = self.default_system_message

        if (self.global_usage.value or 0) > (self.global_limit.value or 0):
            return {
                "response": "Sorry, I'm out of tokens. Please try again later.",
                "usage": 0,
            }
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=history + [{"role": "system", "content": system}],
            temperature=self.openai_params.child("temperature").value or 0.6,
            top_p=self.openai_params.child("top_p").value or 0.9,
            max_tokens=self.openai_params.child("max_tokens").value or 1500,
        )
        # return token usage bundled with response, excluding everything else
        # return response.choices[0].message.content

        self.global_usage.set(
            (self.global_usage.value or 0) + response.usage.total_tokens
        )
        return {
            "response": response.choices[0].message.content,
            "usage": response.usage.total_tokens,
        }

    # listen to all messages
    @commands.Cog.listener("on_message")
    async def on_message(self, message: nextcord.Message):
        if message.author.bot:
            return
        if self.ignored_users[str(message.author.id)]:  # if user is ignored
            return
        print(self.ignored_users.value)

        # check for dms
        if isinstance(message.channel, nextcord.DMChannel):
            # send typing indicator
            async with message.channel.typing():
                pass
            # check if USER is over limit

            if (self.user_usage[str(message.author.id)] or 0) > self.user_limit.value:
                await message.channel.send(
                    f"Sorry <@{message.author.id}>, you've reached your usage limit. Please try again later."
                )
                return

            history = self.dm_chat_history[str(message.author.id)] or []
            system = (
                self.dm_system_messages[str(message.author.id)]
                or self.default_system_message.value
            )

            if len(history) > 10:
                history = history[-10:]
            history.append({"role": "user", "content": message.content})
            response = await self.get_response(history=history, system=system)
            history.append({"role": "assistant", "content": response["response"]})
            self.user_usage[str(message.author.id)] = (
                self.user_usage[str(message.author.id)] or 0
            ) + response["usage"]
            await message.reply(response["response"], mention_author=False)

            self.dm_chat_history[str(message.author.id)] = history

            return

        # it's a guild message
        # send typing indicator

        is_in_bot_channel = (
            str(message.channel.id) in (self.bot_channel[str(message.guild.id)] or [])
        ) or str(message.channel.id) == self.default_bot_channel.value

        if (
            is_in_bot_channel
            or self.bot.user in message.mentions
            or message.reference is not None
            and message.reference.resolved.author == self.bot.user
        ):
            async with message.channel.typing():
                pass
            # check if USER is over limit
            if message.author.id not in (self.user_usage.value or {}).keys():
                self.user_usage[str(message.author.id)] = 0
            if (self.user_usage[str(message.author.id)] or 0) > (
                self.user_limit.value or 0
            ):
                await message.channel.send(
                    f"Sorry <@{message.author.id}>, you've reached your usage limit. Please try again later."
                )
                return

            # check if global over limit performed in get_response
            history = (
                self.guild_chat_history.child(str(message.guild.id))[
                    str(message.channel.id)
                ]
                or []
            )

            system = (
                self.guild_system_messages.child(str(message.guild.id))[
                    str(message.channel.id)
                ]
                or self.default_system_message.value
            )

            if len(history) > 10:
                history = history[-10:]

            history.append({"role": "user", "content": message.content})
            response = await self.get_response(history=history, system=system)
            history.append({"role": "assistant", "content": response["response"]})

            self.user_usage[str(message.author.id)] = (
                self.user_usage[str(message.author.id)] or 0
            ) + response["usage"]

            self.guild_chat_history.child(str(message.guild.id))[
                str(message.channel.id)
            ] = history
            await message.reply(response["response"], mention_author=False)

    @commands.Cog.listener("on_guild_join")
    async def on_guild_join(self, guild: nextcord.Guild):
        # send message to any channel
        await guild.system_channel.send(
            f"""\
Hi! I'm {self.bot.user.mention}. I'm a bot that uses OpenAI's GPT-3 to respond to messages. I'm still in beta, so make sure to send <@892912043240333322> any feedback you have!!\
            """
        )

    @nextcord.slash_command(name="clear", description="Clears context for a channel.")
    async def clear_context(self, interaction: nextcord.Interaction):
        if isinstance(interaction.channel, nextcord.DMChannel):
            self.dm_chat_history[str(interaction.user.id)] = []
            await interaction.response.send_message("Cleared context.")
        else:
            self.guild_chat_history.child(str(interaction.guild.id))[
                str(interaction.channel.id)
            ] = []
            await interaction.response.send_message("Cleared context.")

    @nextcord.slash_command(
        name="gettokens", description="Gets token usage for a user."
    )
    async def get_token_usage(self, interaction: nextcord.Interaction):
        await interaction.response.send_message(
            f"User token usage: {self.user_usage[str(interaction.user.id)]}"
        )

    @nextcord.slash_command(name="ignore", description="Toggles ignore status.")
    async def ignore(self, interaction: nextcord.Interaction):
        self.ignored_users[str(interaction.user.id)] = not self.ignored_users[
            str(interaction.user.id)
        ]
        if self.ignored_users[str(interaction.user.id)]:
            await interaction.response.send_message(
                "You are now ignored.", ephemeral=True
            )
        else:
            await interaction.response.send_message(
                "You are no longer ignored.", ephemeral=True
            )
