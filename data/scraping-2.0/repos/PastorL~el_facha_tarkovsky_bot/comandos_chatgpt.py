import openai
import os
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()
CHATGPT_API_KEY = os.getenv('CHATGPT_API_KEY')

openai.api_key = CHATGPT_API_KEY

@commands.command()
async def chachi(ctx,*,prompt):
    response = chatgpt(prompt)
    await ctx.send(response)

def chatgpt(prompt, model="gpt-3.5-turbo"):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = response.choices[0].text.strip()
    return message

async def setup(bot):
    bot.add_command(chachi)