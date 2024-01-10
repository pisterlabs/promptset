import os
import json
import openai
import discord
import requests
from discord import ui, Embed, app_commands
from discord.ext import commands

with open('config.json') as f:
    config = json.load(f)

openai.api_key = config["openai_token"]


class ai_commands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    async def generate_response(self, ctx, prompt: str, engine: str):
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=300,
            n=1,
            stop=None,
            temperature=0.7,
        )
        response_text = response.choices[0].text.strip()
        if len(response_text) > 1900:
            response_text = response_text[:1900] + "..."

        embed = Embed(title="OpenAI Generator",
                      description="")
        embed.color = 0x3498db
        embed.add_field(name="Prompt",
                        value='```' + prompt + '```',
                        inline=False)
        embed.add_field(name="Response",
                        value='```' + response_text + '```',
                        inline=False)
        await ctx.send(embed=embed)

    @commands.command()
    async def gpt4(self, ctx, *, prompt: str):
        await self.generate_response(ctx, prompt, "text-davinci-002")

    @commands.command()
    async def gpt3(self, ctx, *, prompt: str):
        await self.generate_response(ctx, prompt, "davinci")

    @commands.command()
    async def dalle(self, ctx, *, prompt: str):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }

        data = {
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024"
        }

        response = requests.post(
            "https://api.openai.com/v1/images/generations", headers=headers, json=data)

        if response.status_code == 200:
            image_url = response.json()["data"][0]["url"]
            embed = Embed(title="DALLE Image Generator",
                          description="")
            embed.color = 0x3498db
            embed.add_field(name="Prompt",
                            value='```' + prompt + '```',
                            inline=False)
            print(image_url)
            embed.set_image(url=image_url)
            await ctx.send(embed=embed)
        else:
            await ctx.send(f"Error: Unable to generate image. (Status code: {response.status_code})")


async def setup(bot):
    await bot.add_cog(ai_commands(bot))
