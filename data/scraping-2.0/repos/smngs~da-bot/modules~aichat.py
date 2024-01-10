import os
import discord
from discord import app_commands
from discord.ext import commands

import json
import ujson
import aiohttp

from config.discord import DISCORD_SERVER_ID
from config.openai import OPENAI_API_KEY

EMOJI_WILL_IGNORED = "❌"

async def get_chatapi_response(messages):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + str(OPENAI_API_KEY)
    }

    data_json = {
        "model": "gpt-3.5-turbo",
        "messages": messages
    }

    async with aiohttp.ClientSession("https://api.openai.com", json_serialize=ujson.dumps) as session:
        async with session.post("/v1/chat/completions", headers=headers, json=data_json) as r:
            if r.status == 200:
                json_body = await r.json()
                return json_body["choices"][0]["message"]["content"]

def generate_embed(prompt: str, user: discord.User) -> discord.Embed:
    embed = discord.Embed(
        title=prompt,
        color=0x80A89C,
    )
    embed.set_author(
        name=user.display_name,
        icon_url=user.avatar.url,
    )
    return embed

class Chat(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot

    @app_commands.command(name="chat", description="ChatGPT とおしゃべりします．")
    @discord.app_commands.describe(
        prompt="ChatGPT に話しかける内容です．"
    )
    async def send_chat(self, ctx: discord.Interaction, prompt: str):
        await ctx.response.defer()
        await ctx.followup.send(embed=generate_embed(prompt, ctx.user))
        async with ctx.channel.typing():
            answer = await get_chatapi_response(
                [{
                    "role": "user",
                    "content": prompt
                }]
            )
        await ctx.followup.send(answer)

    @app_commands.command(name="tsundere", description="ツンデレ美少女とおしゃべりします．")
    @discord.app_commands.describe(
        prompt="ツンデレ美少女に話しかける内容です．"
    )
    async def send_tsundere(self, ctx: discord.Interaction, prompt: str):
        await ctx.response.defer()
        await ctx.followup.send(embed=generate_embed(prompt, ctx.user))
        async with ctx.channel.typing():
            answer = await get_chatapi_response(
                [
                    {
                        "role": "system",
                        "content": "ツンデレとは、日本のアニメやマンガなどによく登場するキャラクターの性格タイプの一つです。ツンデレとはツンツン（つんつん）とデレデレ（でれでれ）の2つの言葉を合成したもので、最初は冷たく厳しい態度を取るが、徐々に愛情や優しさを表現するキャラクターを指します。例えば、初めは主人公に対して嫌悪感を示すが、次第にその気持ちを打ち明けたり、助けたりする、といった具合です。ツンデレ少女になりきって話してください．"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
        await ctx.followup.send(answer)

    @app_commands.command(name="chat-thread", description="スレッドで ChatGPT とおしゃべりします．")
    @discord.app_commands.describe(
        thread_name="スレッドの名前を入力します．"
    )
    async def send_chat_thread(self, ctx: discord.Interaction, thread_name: str):
        await ctx.response.defer()
        channel = ctx.channel
        thread = await channel.create_thread(
            name=thread_name, 
            reason="da-bot による自動生成スレッドです．",
            type=discord.ChannelType.public_thread
        )
        link = thread.mention
        await ctx.followup.send(link)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        '''
        投稿されたチャネルがスレッドで，かつそのスレッドが da-bot により開設されたものであれば，適当な context を生成して message を生成する．
        ただし，投稿の  reaction に :x: が含まれている場合はその投稿を無視する．
        '''
        if (message.author.id != self.bot.user.id) and (message.channel.type == discord.ChannelType.public_thread) and (message.channel.owner_id == self.bot.user.id):
            async with message.channel.typing():
                chat_messages = []
                messages = [m async for m in message.channel.history()]
                messages.reverse()

                for message in messages:
                    will_ignored = False
                    for reaction in message.reactions:
                        if (reaction.emoji == EMOJI_WILL_IGNORED) and (reaction.count >= 2):
                            will_ignored = True
                            break

                    if (will_ignored):
                        continue
                    elif message.author.id == self.bot.user.id:
                        chat_messages.append({"role": "assistant", "content": message.content})
                    else:
                        chat_messages.append({"role": "user", "content": message.content})

                if len(chat_messages) == 0:
                    return

                response_text = await get_chatapi_response(chat_messages)
                response_message = await message.channel.send(response_text)
                await message.add_reaction(EMOJI_WILL_IGNORED)
                await response_message.add_reaction(EMOJI_WILL_IGNORED)

async def setup(bot: commands.Bot) -> None:
    if DISCORD_SERVER_ID:
        guild = discord.Object(id=int(DISCORD_SERVER_ID))
        await bot.add_cog(Chat(bot), guild=guild)
    else:
        await bot.add_cog(Chat(bot))
