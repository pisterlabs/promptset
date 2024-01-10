import os
import platform
import re
import sys
from datetime import datetime as dt
from typing import Optional

import openai
from discord import Client, Emoji, Intents, Member, Message, Object, Role, TextChannel, Thread, utils
from discord.app_commands import CommandTree

from action import Action
from botutility import BotUtility
from cronfunction import CronFunction


class MyClient(BotUtility, Client):
    def __init__(self):
        super().__init__(intents=Intents.all())

        self.GUILD: Object = Object(self.get_environvar("GUILDID"))

        self.tree = CommandTree(self)
        self.action: Action = Action()

        self.TEXT_CHANNELS: list[TextChannel]
        self.pins: list[Message] = []

        openai.api_key = self.get_environvar("KEY_OPENAI")

    async def setup_hook(self) -> None:
        self.tree.clear_commands(guild=self.GUILD)
        self.tree.copy_global_to(guild=self.GUILD)

        await self.tree.sync(guild=self.GUILD)

    async def run_cron(self, fname: str, channel: TextChannel) -> None:
        """cron process"""

        # Process
        CronFunction(self.guilds[0])

        await channel.send(f"Start cron: {fname}")
        await eval(f"cf.{fname}(channel)")
        await channel.send(f"End cron: {fname}")

    async def on_ready(self):
        """Awakening of the Unbobo"""

        # Post channel
        channel: Optional[TextChannel] = utils.find(lambda c: c.name == "開発室", self.guilds[0].text_channels)

        # Error handling
        if not channel:
            return

        # Post
        reply: str = "----\n"

        # update time
        update: str = str(dt.fromtimestamp(os.path.getmtime(os.path.abspath(__file__))))
        reply += f"Update: {update}\n\n"

        # platform
        for k, v in platform.uname()._asdict().items():
            reply += f"{k}: {v}\n"

        await channel.send(reply)

        # cron process
        if len(sys.argv) > 1:
            function_name: str = sys.argv[1]
            await self.run_cron(function_name, channel)

            await self.close()
            exit()

        # commands cache
        self.TEXT_CHANNELS = self.guilds[0].text_channels
        for ch in self.TEXT_CHANNELS:
            print(ch.name)
            self.pins += await ch.pins()

    async def chatgpt(self, message: Message):
        escape_content: str = re.sub(r"<@(everyone|here|[!&]?[0-9]{17,20})> ", "", message.content)

        if type(message.channel) is Thread:
            thread: Thread = message.channel

            # chatgpt用にbotが作成したthreadでなければreturn
            if thread.owner != self.user:
                return

            messages: list[dict[str, str]] = []
            async for mes in thread.history():
                role: str = "assistant" if mes.author == self.user else "user"

                messages.append({"role": role, "content": mes.content})

            messages = messages[:200][::-1]

        else:
            thread: Thread = await message.create_thread(name=escape_content[:20])

            messages: list[dict[str, str]] = [{"role": "user", "content": escape_content}]

        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

        reply = "\n".join([choice["message"]["content"] for choice in response["choices"]])
        await thread.send(reply)

    async def check_trigger(self, message: Message):
        """Check trigger on message"""
        for (trigger, pattern) in self.action.PATTERNS.items():
            if re.findall(pattern, message.content):
                await eval(f"self.action.{trigger}(message)")

    async def on_message(self, message: Message):
        print(message.content)

        if self.user in message.mentions or type(message.channel) is Thread:
            if message.author == self.user:
                return

            print("bot mention")
            await self.chatgpt(message)  # 返信する非同期関数を実行
            return

        await self.check_trigger(message)

    async def on_member_update(self, before: Member, after: Member):

        # Post channel
        channel: Optional[TextChannel] = utils.find(lambda c: c.name == "random", self.guilds[0].text_channels)

        # Error handling
        if not channel:
            return

        # updated role
        change_role: list[Role] = list(set(before.roles) ^ set(after.roles))  # Role difference list
        for role in change_role:
            if role in before.roles:  # DELETE role
                reply: str = f"{after.display_name}: {role.name}じゃなくなりました"
            else:  # ADD role
                reply: str = f"{after.display_name}: {role.name}になりました"

            # POST
            await channel.send(reply)

    async def on_guild_emojis_update(self, before: list[Emoji], after: list[Emoji]) -> None:

        # Post channel
        channel: Optional[TextChannel] = utils.find(lambda c: c.name == "開発室", self.guilds[0].text_channels)

        if not channel:
            return

        # Emoji difference list
        change_emoji: list[Emoji] = list(set(before) ^ set(after))
        for emoji in change_emoji:
            if emoji in before:  # DELETE emoji
                reply: str = f"DELETE: {str(emoji)}"
            else:  # ADD emoji
                reply: str = f"ADD: {str(emoji)}"

            # POST
            await channel.send(reply)
