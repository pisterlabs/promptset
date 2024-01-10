import time
import re
import os
import asyncio
import requests
from eden.client import Client
import discord
from discord.ext import commands
from marsbots_core.models import ChatMessage
from marsbots_core.programs.lm import complete_text
from marsbots_core.resources.language_models import OpenAIGPT3LanguageModel
from marsbots_core.resources.discord_utils import (
    is_mentioned,
    remove_role_mentions,
    replace_bot_mention,
    replace_mentions_with_usernames,
    role_is_mentioned,
)
from . import settings, prompts, channels, utils

client = Client(
    url = settings.EDEN_SERVER_URL, 
    username = 'mars_college'
)

class ClipArt(commands.Cog):
    def __init__(self, bot: commands.bot) -> None:
        self.bot = bot
        self.language_model = OpenAIGPT3LanguageModel(
            engine=settings.GPT3_ENGINE,
            temperature=settings.GPT3_TEMPERATURE,
            frequency_penalty=settings.GPT3_FREQUENCY_PENALTY,
            presence_penalty=settings.GPT3_PRESENCE_PENALTY,
        )

    @commands.Cog.listener("on_message")
    async def on_message(self, message: discord.Message) -> None:
        dm = isinstance(message.channel, discord.channel.DMChannel)
        dm_allowed = dm and message.author.id in channels.ALLOWED_DM_USERS
        if (
            (
                is_mentioned(message, self.bot.user)
                or dm_allowed
            )
            and message.author.id != self.bot.user.id
            and (dm_allowed or message.channel.id in channels.ALLOWED_CHANNELS)
        ):
            ctx = await self.bot.get_context(message)
            prompt = self.message_preprocessor(message).strip()
            url = re.search("(?P<url>https?://[^\s]+)", prompt)
            
            if url:
                url = url.group("url")
            
            if message.attachments:
                url = message.attachments[0].url
            
            if url:
                prompt = prompt.replace(url, "").strip()
                config = {"text_input": prompt, "image_url": url}
            else:
                config = {"text_input": prompt}
            
            result = client.run(config)
                
            if "token" not in result:
                async with ctx.channel.typing():
                    await message.reply("Something went wrong :(")
                return

            token = result["token"]

            async with ctx.channel.typing():

                completion = await complete_text(
                    self.language_model, prompts.start_prompt, max_tokens=100, stop=["\n", "\n\n"],
                    use_content_filter=True
                )
                await message.reply(completion)

                finished = False
                while not finished:
                    result = client.fetch(token = token)
                    status = result["status"]["status"]
                    if status == "complete":
                        filepath = '_last_image_.png'
                        output_img = result['output']['creation']
                        output_img.save(filepath)
                        finished = True
                        local_file = discord.File(filepath, filename=filepath)
                        completion = await complete_text(
                            self.language_model, prompts.stop_prompt, max_tokens=100, stop=["\n", "\n\n"],
                            use_content_filter=True
                        )
                        result_string = f'{completion}\n\n**{prompt}**:'
                        await message.reply(result_string, file=local_file)
                        
                    elif status == "failed":
                        finished = True
                        await message.reply("Something went wrong :(")
                    else:
                        #time.sleep(2)
                        asyncio.sleep(self.interval)

    def format_prompt(self, messages):
        last_message_content = replace_bot_mention(messages[-1].content).strip()
        messages_content = self.format_messages(messages)
        topic_idx = self.get_similar_topic_idx(last_message_content)
        topic_prelude = prompts.topics[topic_idx]["prelude"]
        prompt = (
            self.format_prompt_messages(prompts.prelude)
            + "\n"
            + self.format_prompt_messages(topic_prelude)
            + "\n"
            + messages_content
            + "\n"
            + "**["
            + self.bot.user.name
            + "]**:"
        )
        return prompt

    def format_prompt_messages(self, messages):
        return "\n".join(
            [
                "**[%s]**: %s" % (message["sender"], message["message"])
                for message in messages
            ]
        )

    def format_messages(self, messages_content):
        return "\n".join(
            [
                str(
                    ChatMessage(
                        self.message_preprocessor(message_content),
                        self.get_sender(message_content),
                        deliniator_left="**[",
                        deliniator_right="]**:",
                    )
                )
                for message_content in messages_content
            ]
        )

    def get_similar_topic_idx(self, query: str) -> int:
        docs = [topic["document"] for topic in prompts.topics]
        res = self.language_model.document_search(docs, query)
        return self.language_model.most_similar_doc_idx(res)

    def message_preprocessor(self, message: discord.Message) -> str:
        message_content = replace_bot_mention(message.content, only_first=True)
        message_content = replace_mentions_with_usernames(
            message_content, message.mentions
        )
        message_content = remove_role_mentions(message_content)
        message_content = message_content.strip()
        return message_content

    def get_sender(self, message: discord.Message) -> str:
        if message.author.id == self.bot.user.id:
            return self.bot.user.name
        else:
            return f"{utils.get_nick(message.author)}"


def setup(bot: commands.Bot) -> None:
    bot.add_cog(ClipArt(bot))
