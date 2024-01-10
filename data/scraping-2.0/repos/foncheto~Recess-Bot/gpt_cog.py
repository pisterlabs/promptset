import discord
from discord.ext import commands
import json
from langchain import PromptTemplate
from langchain.llms import OpenAI

# Load the Discord API key from credentials.json
with open("credentials.json") as file:
    credentials = json.load(file)
    TOKEN = credentials["openai_api"]

openai = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=TOKEN, temperature=0.8)

template = """ Answer the question below and if the question mentions it return code.
Question: {question}
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=template,
)


class gpt_cog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(
        name="gpt", help="Generates text from a prompt using OpenAI's GPT-3 API."
    )
    async def gpt(self, ctx, *args):
        question = " ".join(args)
        print(question)
        output = openai(prompt_template.format(question=question))
        await ctx.send(output)
