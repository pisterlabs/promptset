import discord
from discord.ext import commands
import openai

# Defina as intenções do bot
intents = discord.Intents.default()
intents.typing = False  # Desabilita a intenção de digitar
intents.presences = False  # Desabilita a intenção de presença

# Configurações do bot Discord com intenções
bot = commands.Bot(command_prefix='.', intents=intents)

# Configuração da API do GPT-3
openai.api_key = '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'

@bot.event
async def on_ready():
    print(f'Bot está online como {bot.user.name}')

@bot.command()
async def iniciar(ctx):
    if not ctx.author.bot:
        overwrites = {
            ctx.guild.default_role: discord.PermissionOverwrite(read_messages=False),
            ctx.author: discord.PermissionOverwrite(read_messages=True),
            bot.user: discord.PermissionOverwrite(read_messages=True)
        }
        channel = await ctx.guild.create_text_channel(f'canal-privado-{ctx.author.display_name}', overwrites=overwrites)
        await ctx.send(f'Canal privado criado: {channel.mention}')

@bot.command()
async def status(ctx):
    # Comando para verificar se o bot está ativo no servidor
    await ctx.send("O bot está ativo e pronto para uso!")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Processa os comandos em qualquer canal
    await bot.process_commands(message)

    if isinstance(message.channel, discord.TextChannel):
        # Aqui você integra o GPT-3 para responder às mensagens no canal
        try:
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=message.content,
                max_tokens=50  # Você pode ajustar o número de tokens para controlar o tamanho da resposta
            )
            bot_response = response.choices[0].text
            await message.channel.send(bot_response)
        except Exception as e:
            print(e)

# Substitua 'SEU_TOKEN_DO_DISCORD' pelo token do seu bot Discord
bot.run('MTE1NzUyMTUzMjcxNzkxMjA5NQ.GIvwOB.KCxvbhHV_33ejIQ1opFFMqQGtb13sBseZrE2EA')
