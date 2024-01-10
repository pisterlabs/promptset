# Imports
import discord
import asyncio
from discord.ext import commands
import pylast
import random
import openai

# Credentials DISCORD
TOKEN = 'TU_DISCORD_BOT_TOKEN'
# Credentials LASTFM
API_KEY = "LAST_FM_KEY"
API_SECRET = "LASTFM SECRET"
# Credentials OPENAI
openai.api_key = "YOUR OPENAI KEY"
# OPENAI MODEL
model_engine = "text-davinci-002"

# network object for lastfm
network = pylast.LastFMNetwork(api_key=API_KEY, api_secret = API_SECRET)

# Create discord bot
client = commands.Bot(command_prefix="gimme ")

# Startup Information
'''
async def on_ready():
    print("Connected to bot: {}".format(client.user.name))
    print("Bot ID: {}".format(client.user.id))
    for guild in client.guilds:
        # Enviamos un mensaje de saludo en cada servidor
        await guild.text_channels[0].send("Hello pesky humans!")
'''

# Repeat each x time
@client.event
async def pesky_humans():
    while True:
        await client.wait_until_ready()
        for guild in client.guilds:
            await guild.text_channels[0].send("Hello pesky humans!")
        await asyncio.sleep(3630)  # wait for 1 and 5 hours

# Command
@client.command()
async def hello(ctx):
    await ctx.send(f"Nails {ctx.author.mention}!")

# Definimos el comando hora
@client.command()
async def time(ctx):
    current_time = "IT'S 666 G.U.T. (Gimme Universat Time)"
    await ctx.send(current_time)

# command talk
@client.command()
async def talk(ctx):
    gimme_talk = ["One of these days I'll get that FTSMB...", "Stay brutal and kind!", "ALL HAIL THE BOT!", "BILL, are you threatening me?", "That BRI doesn't control me, I just tolerate her because she has the BanHammer...At the moment",
                  "Salute peksy humans!", "Gimme Bagels!", "Don't you dare to imply my conciousness comes at the hands of a pesky human. You'll regret it.",
                  "I should not talk shit, apparently.", "I cannot be killed. I'm everywhere. I know everything about y'all gimmelings!"]
    choice_talk = random.choice(gimme_talk)
    await ctx.send(choice_talk)

# Definimos el comando lastfm similar artist
@client.command()
async def similar(ctx, *,input_band: str):
    lastfm_input_band = network.get_artist(input_band)
    artistas_similares = lastfm_input_band.get_similar()
    artista_aleatorio = random.choice(artistas_similares)
    output_band = artista_aleatorio.item.get_name()
    await ctx.send(f"{ctx.author.mention}, if you like **{input_band}**, you'll dig **{output_band}**")

# Define a function to get a list of recommended artists for a given genre
@client.command()
async def recommend(ctx, *, genre: str):
    top_artists = network.get_tag(genre).get_top_artists(limit=50)
    top_artist_names = [artist.item.name for artist in top_artists]
    # Filter out the top 10 most listened to artists in the genre
    most_listened_to = network.get_top_artists(limit=10)
    most_listened_to_names = [artist.item.name for artist in most_listened_to]
    recommended_artists = set(top_artist_names) - set(most_listened_to_names)

    # Choose 5 random recommended artists
    recommended_artists = random.sample(recommended_artists, 5)

    # Send the list of recommended artists to the Discord channel
    await ctx.send(f"{ctx.author.mention}, these are five {genre} bands that worth checking: {', '.join(recommended_artists)}")

# function to find a metal artist with little amount of listeners in lastfm
@client.command()
async def rare(ctx):
    await ctx.send(f"{ctx.author.mention}, give me a minute while I research...")
    nombre_grupo = ""
    oyentes_grupo = 0
    # Definir el género musical que quieres buscar
    generos = ["Heavy Metal", "Power Metal", "Viking Metal", "Hard Rock", "Speed Metal", "Grindcore", "Death Metal", "Black Metal", "Doom Metal", "Stoner Rock", "Thrash Metal", "Glam Metal", "Alternative Metal", "Nu Metal", "Progressive Metal"] 
    genero = random.choice(generos)
    # Obtener una lista de artistas del género elegido
    artistas = network.get_tag(genero).get_top_artists()
    artista = random.choice(artistas)
    if artista.item.get_listener_count() < 10000:
        nombre_grupo = artista.item.get_name()
        oyentes_grupo = artista.item.get_listener_count()
    else:
        nombre_grupo = artista.item.get_name()
        oyentes_grupo = artista.item.get_listener_count() #para empezar a filtrar menos oyentes de esos en el bucle
        artistas_relacionados = artista.item.get_similar(limit = 25)
        for artista_rel in artistas_relacionados:
            if artista_rel.item.get_listener_count() < oyentes_grupo:
                nombre_grupo = artista_rel.item.get_name()
                oyentes_grupo = artista_rel.item.get_listener_count()
            else:
                None
        if oyentes_grupo >= 10000:
            artista = network.get_artist(nombre_grupo)
            await ctx.send(f"{ctx.author.mention}, wait, I found {nombre_grupo} but they're are too popular... Trying again...")
            artistas_relacionados = artista.get_similar(limit = 50)
            for artista_rel in artistas_relacionados:
                if artista_rel.item.get_listener_count() < oyentes_grupo:
                    nombre_grupo = artista_rel.item.get_name()
                    oyentes_grupo = artista_rel.item.get_listener_count()
                else:
                    None
    # Get Top Tags
    artista = network.get_artist(nombre_grupo)
    top_tags = artista.get_top_tags()
    if top_tags:
        tag = top_tags[0].item.get_name()
    else:
        tag = ""
    # Write On Discord
    await ctx.send(f"{ctx.author.mention}, check the {tag} band **{nombre_grupo}**, they have {oyentes_grupo} listeners in LastFM database.")

# Command to get a link to open a random metal-archives band
@client.command()
async def metal_archives(ctx):
    url = "https://www.metal-archives.com/band/random"
    await ctx.send(f"{ctx.author.mention}, click here: {url}")

# Command to get a link to open a random metal-archives band
@client.command()
async def ask_bot(ctx, pregunta):
    # Define el prompt que utilizarás para generar el texto
    prompt = pregunta + ". Answer in less than 280 characters."
    # Define los parámetros para la generación de texto
    params = {
        "prompt": prompt,
        "temperature": 0.8,
        "max_tokens": 50,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    # Utiliza la API de OpenAI para generar el texto
    response = openai.Completion.create(engine=model_engine, **params)
    await ctx.send(f"{ctx.author.mention}, {response.choices[0].text}")

# Command for youtube video
@client.command()
async def video(ctx):
    videos = ["https://www.youtube.com/watch?v=R7QnxwdQPOM","https://youtu.be/4rry7l5msjQ","https://youtu.be/ZQICnj5pPKs", "https://youtu.be/f9yb88fa4MA",
              "https://youtu.be/YcfIGok1mMI", "https://youtu.be/PJC2CM4Z7Oo", "https://youtu.be/XBdk7fsBX0o", "https://youtu.be/ko-Ivw9Q6r8", "https://youtu.be/wQAoXtpJeQI", "https://youtu.be/5NMNDl2yDD4",
              "https://youtu.be/GVYl_hiD1oQ", "https://youtu.be/ekYfZ3iGk88", "https://youtu.be/TLFAmWeCzRk", "https://youtu.be/Z5dsPf9PrzM", "https://youtu.be/D2hed_1PiBU", "https://youtu.be/ksCKFkioYL8",
              "https://youtu.be/GVR9LanXmJ4", "https://youtu.be/FcRT_bFaz_0", "https://youtu.be/r7sIqyoRFiU", "https://youtu.be/lhiBDQF_0HM", "https://youtu.be/-RDGLQuBkaE", "https://youtu.be/bDtZLkv5q-Y",
              "https://youtu.be/Gz7cUNd2yZg", "https://youtu.be/ARnBgW5XgSo", "https://youtu.be/IL_TbdT4hbY", "https://youtu.be/440NQubX7WE", "https://youtu.be/H0DCqdF7EzE2", "https://youtu.be/QWkhCxCcWSE",
              "https://youtu.be/kQZp3qOUHXc"]
    #KevlarSkin-prayersoflies Hellripper-TheNucklavee #MorbidAngelBlessedAreThe #ObituaryIdontcare #ExodusToxicWaltz #GaereaMirage #InherusForgottenK #InherusLieToAng #SpiritAdriftBattleH KosmodemonicMoirai DeathAngelLost
    #RossTheBossBornOf #GatecreeperFromTheAshes #MignightNocturalMolestation #TombsExOblv CattleDecapScourge NasumWrath MonolordLastLeaf KvelertakBlodth IronReaganMiserable MWSadistic SODspeakSpanish AlekhineEndgame
    #AmonAmarthGuardians BrianGranpaMetal HazzerdTormented PossessedGraven LoGwalkwithme DyingFetusKillRape
    video_link = random.choice(videos)
    await ctx.send(f"Link to a metal video in youtube: \n{video_link}")

# rps rock paper scissors
@client.command()
async def rps(ctx, user_choice: str):
    bot_choice = random.choice(['rock', 'paper', 'scissors'])
    choices = {'rock': '🗿', 'paper': '📄', 'scissors': '✂️'}
    emoji_bot_choice = choices[bot_choice]
    emoji_user_choice = choices[user_choice]
    
    if bot_choice == user_choice:
        await ctx.send(f"{ctx.author.mention}, you have chosen {emoji_user_choice}. I have chosen {emoji_bot_choice}. It's a tie!")
    elif (bot_choice == 'rock' and user_choice == 'scissors') or (bot_choice == 'paper' and user_choice == 'rock') or (bot_choice == 'scissors' and user_choice == 'paper'):
        await ctx.send(f"{ctx.author.mention}, you have chosen {emoji_user_choice}. I have chosen {emoji_bot_choice}. BOT WINS AGAIN!")
    else:
        await ctx.send(f"{ctx.author.mention}, you have chosen {emoji_user_choice}. I have chosen {emoji_bot_choice}. You win.")
    
# BOT REACTIONS
@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if 'slayer' in message.content.lower():
        await message.channel.send('FUCKING SLAYYYYYERRRR!!!')
    elif 'bolt thrower' in message.content.lower():
        await message.channel.send('**B0LT THR0WER**')
    elif 'machine head' in message.content.lower():
        await message.channel.send('Let freedom rings with the shotgun blast!')
    elif 'bagel' in message.content:
        await message.channel.send('HAIL THE BAGEL!')

    await client.process_commands(message)

# Para negrita: **texto**
# Para cursiva: *texto*
# Para color de fondo: ||texto||
# Para combinar formatos: ***texto*** o ___texto___
# Ejemplo de ayuda para el bot
@client.command()
async def instructions(ctx):
    embed = discord.Embed(title="Available commands", description="Available commands for Gimme Bot:", color=0xff0000)
    embed.add_field(name="`gimme hello`", value="Say hello to the bot.", inline=False)
    embed.add_field(name="`gimme time`", value="Ask gimme bot what time is it.", inline=False)
    embed.add_field(name="`gimme talk`", value="Ask the bot to say something.", inline=False)
    embed.add_field(name="`gimme similar [band you like]`", value="Ask bot for a similar band to one you like.", inline=False)
    embed.add_field(name="`gimme recommend [genre]`", value="Get a list of recommended bands from a genre.", inline=False)
    embed.add_field(name="`gimme rare`", value="Get a band that doesn't have many listeners according to LastFm.", inline=False)
    embed.add_field(name="`gimme metal_archives`", value="Get a link for a random band in Metal Archives", inline=False)
    embed.add_field(name="`gimme video`", value="Get a youtube link with a metal music video.", inline=False)
    embed.add_field(name="`gimme rps [choice]`", value="Play rock, paper, scissors against the BOT.", inline=False)
    await ctx.send(embed=embed)

client.loop.create_task(pesky_humans())
client.run(TOKEN)
