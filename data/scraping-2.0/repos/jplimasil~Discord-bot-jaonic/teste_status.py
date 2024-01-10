import discord
from discord.ext import commands
from openai import AsyncOpenAI, RateLimitError
from dotenv import load_dotenv
import os
import asyncio
import json

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

async def gerar_resposta(ctx, message):
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    try:
        completion = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": message}]
        )
        return completion.choices[0].message['content']
    except RateLimitError:
        await ctx.send("Ola gatinha")

@bot.event
async def on_ready():
    print(f'Bot está conectado como {bot.user}')

@bot.command()
async def teste(ctx, *, message: str):
    gpt_answer = await gerar_resposta(ctx, message)
    await ctx.send(gpt_answer)
    

with open('status_data.json', 'r', encoding='utf-8') as file:
    status_data = json.load(file)

status_dict = status_data['status']
status_synonyms = status_data['synonyms']

@bot.command()
async def status(ctx, *, stat: str = None):
    if stat is None:
        status_list = "\n".join([f"({index + 1}) {status}" for index, status in enumerate(status_dict.keys())])
        await ctx.send("Status disponíveis:\n" + status_list)
        return
    
    stat = stat.lower()
    found = False
    for status, synonyms in status_synonyms.items():
        if stat in synonyms:
            stat = status
            found = True
            break
    
    if stat in status_dict:
        if stat in status_dict and stat.isdigit() and status_dict[stat].get(stat):
            ability = status_dict[stat][stat]
            await ctx.send(f"**{ability['nome']}**:\n{ability['descricao']}")
        else:
            await ctx.send("\n\n".join([f"**{s} pontos:**\n{status_dict[stat][s]['nome']} - {status_dict[stat][s]['descricao']}" for s in status_dict[stat]]))
    elif stat[:-2] in status_dict and stat[-2:] in status_dict[stat[:-2]]:
        await ctx.send(status_dict[stat[:-2]][stat[-2:]]['nome'] + " - " + status_dict[stat[:-2]][stat[-2:]]['descricao'])
    elif not found:
        await ctx.send("Status não encontrado.")

bot.run(os.getenv('DISCORD_TOKEN'))
