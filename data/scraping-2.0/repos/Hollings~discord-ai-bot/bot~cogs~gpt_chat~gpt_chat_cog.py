import asyncio
import datetime
import json
import random
import re
from pathlib import Path

import openai
from discord import Message
from discord.ext import commands
from discord.ext.commands import Cog, Context

from cogs.gpt_chat.system_prompt import SystemPrompt
from peewee import fn


async def setup(bot):
    gpt_chat = GptChat(bot)


class GptChat(commands.Cog):

    def __init__(self, bot):
        self.bot = bot
        bot.db.create_tables([SystemPrompt])
        self.set_system_prompt()
        # load the "systemprompt.txt" file into the system prompt config
        self.bot.config["SYSTEM_PROMPT"] = open("systemprompt.txt", "r").read()
        print("GPT Chat cog init")

    def set_new_random_system_prompt(self):
        # get a random inactive system prompt
        system_prompt = SystemPrompt.select().where(SystemPrompt.active_on.is_null(True)).order_by(fn.Random()).limit(1).get()
        # set it active
        system_prompt.set_active()
        self.bot.config["SYSTEM_PROMPT"] = system_prompt.content

    def set_system_prompt(self, system_prompt=None):
        # if none is sent, get the current active one
        if system_prompt is None:
            try:
                system_prompt = SystemPrompt.get(SystemPrompt.active_on.is_null(False)).content
            except SystemPrompt.DoesNotExist:
                SystemPrompt.create(content="").set_active()
                system_prompt = ""

        # if one is set, create a new one and set it active, then set the old one to inactive
        else:
            SystemPrompt.create(content=system_prompt).set_active()

        # set the config to be used elsewhere
        self.bot.config["SYSTEM_PROMPT"] = system_prompt

    def check_system_prompt_last_set_time(self):
        # if the system prompt was set more than an hour ago, set a new one
        system_prompt = SystemPrompt.get(SystemPrompt.active_on.is_null(False))
        if system_prompt.active_on < datetime.datetime.now() - datetime.timedelta(hours=1):
            self.set_new_random_system_prompt()

    # on message
    @Cog.listener("on_message")
    async def on_message(self, message):
        if message.channel.id != int(self.bot.config["GPT_CHAT_CHANNEL_ID"]) and message.channel.id != int(self.bot.config["DEBUG_CHANNEL_ID"]):
            return

        if message.content.startswith("!*"):
            return

        if message.author.bot:
            return

        if message.content == "!system":
            await message.channel.send("Current System Prompt: ```" + self.bot.config.get("SYSTEM_PROMPT", "(none)") + "```")
            return
        if message.content == "!system reset" or message.content == "!system clear":
            self.set_system_prompt("")
            return
        if message.content.startswith("!system "):
            system_prompt = message.content.replace("!system", "").strip()
            if system_prompt == "roll":
                self.set_new_random_system_prompt()
            else:
                self.set_system_prompt(system_prompt)
            await message.channel.send("Current System Prompt: ```" + self.bot.config.get("SYSTEM_PROMPT", "(none)") + "```")
            return

        if len(message.attachments) > 0:
            # let the gpt-v cog handle this
            return

        self.check_system_prompt_last_set_time()

        ctx = await self.bot.get_context(message)

        typing_task = asyncio.create_task(self.send_typing_indicator_delayed(ctx))
        content = await self.get_gpt_chat_response(message)
        typing_task.cancel()
        if content:
            await message.channel.send(content)


    async def send_typing_indicator_delayed(self, ctx: Context):
        timer = asyncio.sleep(2)
        await timer
        try:
            for i in range(15):
                async with ctx.channel.typing():
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def cog_load(self):
        print("GPT Chat cog loaded")

    async def get_gpt_chat_response(self, message: Message):
        try:
            openai.api_key = self.bot.config['OPENAI_API_KEY']

            messages = [message async for message in message.channel.history(limit=15)]
            formatted_messages = self.format_chat_history(messages, self.bot.config.get("SYSTEM_PROMPT", ""))
            # Make the API request
            response = openai.chat.completions.create(
                response_format={"type": "json_object"},
                model="gpt-4-1106-preview",
                messages=formatted_messages,
                max_tokens=800,
            )
            json_data = json.loads(response.choices[0].message.content)
            try:
                reaction_coroutine = message.add_reaction(json_data["emoji"])
            except Exception as e:
                reaction_coroutine = None
                pass

            await message.channel.send(json_data["content"][:1999])

            if reaction_coroutine:
                await reaction_coroutine
        except Exception as e:
            await message.channel.send("Something went wrong. Please try again later. " + str(e))

    def format_chat_history(self, messages, system_prompt="") -> list:
        formatted_messages = [{"role": "system",
                               "content": "Respond as you normally would, but in the following JSON format: {'emoji': '✅', 'content': 'your message'} the emoji key is anything that you want it to be. Use it to convey emotion, confusion, or anything else. Leave this blank unless you feel its absolutely relevant and necessary. There are multiple users in this chat, the speaker will be identified before each of their message content. Don't copy the 'message author' type formatting in your response. just reply normally.\n\n IMPORTANT Follow these instructions exactly: " + system_prompt}]
        # Take the last 10 messages from the chat history
        total_chars = 0
        for message in messages:
            total_chars += len(message.content)
            if total_chars > 5000:
                break
            role = "assistant" if message.author.bot else "user"
            member = message.guild.get_member(message.author.id)
            nickname = member.nick if member else message.author.name

            formatted_messages.insert(0, {"role": role,
                                          "content": f"(message author: '{nickname}') {message.content}"})
        return formatted_messages
