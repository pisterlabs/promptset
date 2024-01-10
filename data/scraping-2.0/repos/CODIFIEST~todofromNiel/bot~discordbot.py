# imports
import openai
import discord
from discord.ext import commands

# code
bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())
openai.api_key = "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"

@bot.event
async def on_ready():
    print("Bot is ready.")

@bot.command(name='gpt', help='Generate response using GPT-3')
async def generate_response(ctx, *, prompt):
    completions = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = completions.choices[0].text
    await ctx.send(message.strip())

bot.run("MTA2Mjc5NDI5NjQ0NjY4MTEyOQ.G7ivaM.ex83zValOsP3vW4pM6x8yVBKOy7Kmf9q7T6uUk")
