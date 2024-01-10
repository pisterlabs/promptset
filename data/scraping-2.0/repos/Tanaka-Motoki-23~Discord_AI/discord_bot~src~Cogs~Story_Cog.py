import os

import discord
import openai
from discord.ext import commands

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_ORGANIZATION = os.environ.get("OPENAI_ORGANIZATION")

"""
タイトルと主人公を入力すると、それに合わせた小説を生成する。
モデルはGPT-3を利用し、Few－Shotによって出力をコントロールする。
推論はOpenAI APIを利用して行う。
see:https://openai.com/blog/openai-api/
"""


def setup(bot):
    bot.add_cog(Story_Cog(bot))


class Story_Cog(commands.Cog):
    # 接続時の初期設定
    def __init__(self, bot):
        self.bot = bot
        openai.api_key = OPENAI_API_KEY
        openai.organization = OPENAI_ORGANIZATION

    @commands.Cog.listener()
    async def on_ready(self):
        print("Story ready.")

    # STORYコマンド
    @commands.command(name="STORY")
    async def story(self, ctx, title, feature="私", length="middle"):
        try:
            color = discord.Color.random()
            embed = discord.Embed(title=f"タイトル「{title}」", description="生成中...", color=color)
            embed_message = await ctx.channel.send(embed=embed)
            story = self.request_story(title, feature, length)
            story = self.cut_end(story, end_figs=["。", "」"])
            new_embed = discord.Embed(title=f"タイトル「{title}」", description=story, color=color)
            await embed_message.edit(embed=new_embed)
        except Exception:
            new_embed = discord.Embed(
                title=f"タイトル「{title}」", description="うまく生成できませんでした。", color=color
            )
            await embed_message.edit(embed=new_embed)

    @story.error
    async def story_error(self, ctx, error):
        if isinstance(error, commands.BadArgument):
            await ctx.channel.send('`STORY "タイトル" 文章長{short, middle, long, verylong}` を利用出来ます！')

    def request_story(self, title, feature, length):
        if length == "short":
            max_tokens = 200
        elif length == "middle":
            max_tokens = 500
        elif length == "long":
            max_tokens = 700
        elif length == "verylong":
            max_tokens = 800
        else:
            max_tokens = 500

        prompt = f"この物語はフィクションです。\nタイトル「{title}」 制作 {feature}委員会\n{feature}はある朝、"
        print(prompt)
        # 推論
        response = openai.Completion.create(
            engine="davinci", prompt=prompt, max_tokens=max_tokens, temperature=0.7, top_p=1
        )
        story = str(response["choices"][0]["text"])
        print(story)
        return story

    def cut_end(self, text, end_figs="。"):
        fig_index = list()
        for end_fig in end_figs:
            fig_index.append(text.rfind(end_fig))

        if max(fig_index) == -1:
            pass
        else:
            text = text[: max(fig_index) + 1]
        return text
