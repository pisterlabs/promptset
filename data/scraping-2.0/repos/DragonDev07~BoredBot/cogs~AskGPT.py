import openai
import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')

class AskGPT(commands.Cog):
    def __init__(self, client):
        self.client = client

    # Log that the Cog has been loaded
    @commands.Cog.listener()
    async def on_ready(self):
        print("The 'AskGPT' cog has been loaded")

    # Propagate the error to the global error handler
    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):
        await self.bot.on_command_error(ctx, error)

    # AskGPT command which calls the openai api to ask chatgpt a given prompt
    @commands.command()
    async def askgpt(self, ctx, *, prompt:str=None):
        embed = discord.Embed(title = "AskGPT", colour = discord.Colour.blurple())
        if(prompt == None):
            embed.add_field(name="Error", value="Error, you must provide a prompt.")
        else:
            embed.add_field(name="Prompt", value=f"{prompt}")
            embed.add_field(name="Response", value=f"{callGPT(prompt)}")
        await ctx.send(embed = embed)
        print(f"The 'askGPT' command was run by {ctx.message.author}")

def callGPT(prompt):
    print("Running 'callGPT'")
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=2048,
        n=1,
        stop=None,
        temperature=0.5,
    )
    print(completions.choices[0].text)
    return completions.choices[0].text

async def setup(client):
    await client.add_cog(AskGPT(client))