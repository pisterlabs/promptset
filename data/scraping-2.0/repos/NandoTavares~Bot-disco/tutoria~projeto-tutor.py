import discord
from discord.ext import commands
import openai
from dotenv import load_dotenv
import os

load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True

bot = commands.Bot(command_prefix="!", intents=intents)

openai.api_key = OPENAI_API_KEY

async def buscar_historico_canal(canal, limit=5):
    messages_list = []
    async for message in canal.history(limit=limit):
        role = "user" if message.author.id != bot.user.id else "system"
        content = message.content
        messages_list.append(f"{role}: {content}")

    messages_list.reverse()
    return messages_list

def ask_gpt(mensagens):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for mensagem in mensagens:
        role, content = mensagem.split(": ", 1)
        messages.append({"role": role, "content": content})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )

    resposta = response['choices'][0]['message']['content'].strip()
    return resposta

@bot.event
async def on_ready():
    print(f"A {bot.user.name} ficou ligado!")

@bot.event
async def on_message(message):
    try:
        if message.author.bot:
            return

        async with message.channel.typing():
            # Verificar se a mensagem começa com '\'
            if message.content.startswith("\\"):
                mensagens = await buscar_historico_canal(message.channel)
                resposta = ask_gpt(mensagens)

                # Verificar se a resposta não está vazia
                if resposta.strip():
                    await message.reply(resposta)
                else:
                    print("A resposta está vazia ou contém apenas espaços em branco.")

    except Exception as e:
        print(f"Erro ao processar mensagem: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await bot.process_commands(message)

bot.run(DISCORD_BOT_TOKEN)
