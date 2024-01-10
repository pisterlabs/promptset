import os
import openai
import discord
from discord.ext import commands

bot = commands.Bot(command_prefix='^', intents=discord.Intents.all())

async def translate(text:str,lang:str):
    prompt =(
        "You are going to be a good translator "
        "I need this text precisely in {} trying to keep the same meaning "
        "Translate from [START] to [END]:\n[START]"
    )
    prompt=prompt.format(lang)
    prompt += text + "\n[END]"
        
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=prompt,
        max_tokens=3000,
        temperature=0.25
    )
    return response.choices[0].text.strip()

@bot.command(aliases=['traducir', 'traductor', 'traduce', 'traduccion', 'traducciones', 'translate'])
async def translate_text(ctx, lang:str='es'):
    openai.api_key = os.getenv('OPENAI_API')
    await ctx.send("Ingrese el texto a traducir")
    text=await bot.wait_for('message', check=lambda message: message.author == ctx.author)
    await ctx.send("Traduciendo...")
    await ctx.send(await translate(text.content, lang))