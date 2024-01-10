import discord
import asyncio
import os
import wikipedia
from discord.ext import commands
from random import randint
from time import sleep
from riotwatcher import LolWatcher, ApiError
import openai
import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup

intents = discord.Intents.all()
intents.members = True
intents.typing = True
intents.presences = True

bot = commands.Bot(command_prefix="$", description=":tools:", intents=intents)
openai.api_key = os.environ["OPENAI_API_KEY"]


def search_video_on_youtube(query):
    # Préparer la requête à l'API de recherche YouTube
    YOUTUBE_API_KEY = os.environ["YOUTUBE_API_KEY"]
    request_url = 'https://www.googleapis.com/youtube/v3/search'
    request_params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'key': YOUTUBE_API_KEY
    }

    # Envoyer la requête à l'API de recherche YouTube
    response = requests.get(request_url, params=request_params)
    response_json = response.json()

    # Récupérer l'URL de la première vidéo de la liste de résultats
    first_video = response_json['items'][0]
    video_id = first_video['id']['videoId']
    video_url = f'https://www.youtube.com/watch?v={video_id}'

    return video_url


def rock_paper(p_1, p_2):
    roll = randint(1, 2)
    if roll == 1:
        result = f"{p_1} a Gagné !"
    else:
        result = f"{p_2} a Gagné !"
    return result


@bot.event
async def on_ready():
    print("Bot Ready")


@bot.command()
async def dice(ctx, m=-1):
    if m == -1:
        await ctx.send(f"Dé à 6 faces : {randint(1,6)}")
        return None
    if m != 0:
        await ctx.send(f"Dé à {m} faces : {randint(1,m)}")


@bot.command()
async def opgg(ctx, name):
    summ_name = name.replace(" ", "")
    await ctx.send(f"https://euw.op.gg/summoner/userName={summ_name.lower()}")


@bot.command()
async def duel(ctx, p_1, p_2):
    aff = rock_paper(str(p_1), str(p_2))
    await ctx.send(aff)


@bot.command()
async def champion(ctx):
    champion = ["Aatrox", "Ahri", "Akali", "Alistar", "Amumu", "Anivia", "Annie", "Ashe", "Azir", "Bard", "Blitzcrank", "Brand", "Braum", "Caitlyn", "Cassiopeia", "Cho'Gath", "Corki", "Darius", "Diana", "Dr. Mundo", "Draven", "Ekko", "Elise", "Evelynn", "Ezreal", "Fiddlesticks", "Fiora", "Fizz", "Galio", "Gangplank", "Garen", "Gnar", "Gragas", "Graves", "Hecarim", "Heimerdinger", "Irelia", "Janna", "Jarvan IV", "Jax", "Jayce", "Jinx", "Kalista", "Karma", "Karthus", "Kassadin", "Katarina", "Kayle", "Kennen", "Kha'Zix", "Kindred", "Kog'Maw", "LeBlanc", "Lee Sin", "Leona", "Lissandra", "Lucian", "Lulu", "Lux", "Malphite", "Malzahar", "Maokai", "Master Yi", "Miss Fortune", "Mordekaiser", "Morgana", "Nami", "Nasus", "Nautilus", "Nidalee", "Nocturne", "Nunu", "Olaf", "Orianna", "Pantheon",
                "Poppy", "Quinn", "Rammus", "Rek'Sai", "Renekton", "Rengar", "Riven", "Rumble", "Ryze", "Samira", "Sejuani", "Shaco", "Shen", "Shyvana", "Singed", "Sion", "Sivir", "Skarner", "Sona", "Soraka", "Swain", "Syndra", "Tahm Kench", "Talon", "Taric", "Teemo", "Thresh", "Tristana", "Trundle", "Tryndamere", "Twisted Fate", "Twitch", "Udyr", "Urgot", "Varus", "Vayne", "Veigar", "Vel'Koz", "Vi", "Viktor", "Vladimir", "Volibear", "Warwick", "Wukong", "Xerath", "Xin Zhao", "Yasuo", "Yone", "Yorick", "Zac", "Zed", "Ziggs", "Zilean", "Zyra", "Yuumi", "Seraphine", "Lillia", "Sett", "Aphelios", "Senna", "Qiyana", "Sylas", "Neeko", "Pyke", "Kai'Sa", "Zoe", "Ornn", "Kayn", "Rakan", "Xayah", "Camille", "Ivern", "Kled", "Taliyah", "Aurelion Sol", "Jhin", "Illaoi", "Rell", "Viego", "Gwen"]
    index = randint(0, len(champion)-1)
    aff = champion[index]
    await ctx.send(aff)


@bot.command()
async def role(ctx, role=''):
    dico_roles = {"top": ["Aatrox", "Akali", "Cho'Gath", "Darius", "Dr.Mundo", "Fiora", "Gangplank", "Garen", "Illaoi", "Irelia", "Jax", "Jayce", "Kayle", "Kennen", "Kled", "Lucian", "Malphite", "Maokai", "Mordekaiser", "Nasus", "Poppy", "Quinn", "Renekton", "Riven", "Rumble", "Sett", "Shen", "Singed", "Sion", "Sylas", "Tahm Kench", "Teemo", "Tryndamere", "Urgot", "Vayne", "Vladimir", "Volibear", "Wukong", "Yasuo", "Yone", "Yorick", "Ornn", "Gnar", "Camille", "Gwen"],
                  "jungle": ["Amumu", "Ekko", "Elise", "Evelynn", "Fiddlesticks", "Gragas", "Graves", "Hecarim", "Yvern", "Jarvan IV", "Jax", "Karthus", "Kayn", "Kha'Zix", "Kindred", "Lee Sin", "Lillia", "Maître Yi", "Nidalee", "Nocturne", "Nunu et Willump", "Olaf", "Rammus", "Rek'Sai", "Rengar", "Sejuani", "Sett", "Shaco", "Shyvana", "Skarner", "Sylas", "Taliyah", "Trundle", "Udyr", "Vi", "Volibear", "Warwick", "Xin Zhao", "Zac", "Viego"],
                  "mid": ["Ahri", "Akali", "Anivia", "Annie", "Aurelion Sol", "Azir", "Cassiopeia", "Corki", "Diana", "Ekko", "Fizz", "Galio", "Heimerdinger", "Irelia", "Kassadin", "Katarina", "Leblanc", "Lissandra", "Lucian", "Lux", "Malzahar", "Neeko", "Orianna", "Qiyana", "Ryze", "Seraphine", "Sylas", "Syndra", "Talon", "Twisted Fate", "Veigar", "Viktor", "Vladimir", "Xerath", "Yasuo", "Yone", "Zed", "Ziggs", "Zoé", "Akshan"],
                  "adc": ["Aphelios", "Ashe", "Caitlyn", "Draven", "Ezreal", "Jhin", "Jinx", "Kai'Sa", "Kalista", "Kog'Maw", "Lucian", "Miss Fortune", "Samira", "Senna", "Sivir", "Tristana", "Twitch", "Varus", "Xayah", "Yasuo", "Akshan"],
                  "supp": ["Alistar", "Bard", "Blitzcrank", "Brand", "Braum", "Janna", "Karma", "Leona", "Lulu", "Lux", "Malphite", "Maokai", "Morgana", "Nami", "Nautilus", "Pantheon", "Pyke", "Rakan", "Senna", "Seraphine", "Sett", "Sona", "Soraka", "Swain", "Tahm Kench", "Taric", "Thresh", "Vel'koz", "Xerath", "Yuumi", "Zilean", "Zyra", "Rell"]
                  }
    if role == '':
        await ctx.send(dico_roles.keys()[randint(1, 5)])
        return None
    rand = randint(0, len(dico_roles[role])-1)
    liste = (list(dico_roles[role]))
    aff = liste[rand]
    await ctx.send(aff)


@bot.command()
async def summoner(ctx, name, region):
    """
    region: BR1/EUN1/EUW1/JP1/KR/LA1/LA2/NA1/OC1/TR1/RU
    """
    api_key = os.environ['RIOT_TOKEN']
    watcher = LolWatcher(api_key)
    me = watcher.summoner.by_name(region, name)
    my_ranked_stats = watcher.league.by_summoner(region, me['id'])
    i = 0
    z = 0
    while i < len(my_ranked_stats):
        val = list(my_ranked_stats[i].values())
        j = 0
        while j < len(my_ranked_stats[i]):
            if val[j] == 'RANKED_SOLO_5x5':
                z = i
            j += 1

        i += 1

    solo = my_ranked_stats[z]
    winrate = round(solo["wins"]/(solo["wins"]+solo["losses"])*100)
    tier = solo["tier"]
    rank = solo["rank"]
    wins = solo["wins"]
    losses = solo["losses"]
    nom = solo["summonerName"]
    lp = solo["leaguePoints"]
    aff = "Rank: " + tier + " " + rank + " LP: " + \
        str(lp) + " Winrate: " + str(winrate)+"%" + \
        " W: " + str(wins) + " L: " + str(losses)
    aff2 = "SOLO RANKED informations for " + nom
    await ctx.send(aff2)
    await ctx.send(aff)


@bot.command()
async def wiki(ctx, sub):
    """
    Affiche la page wikipedia passé en paramétre
    """
    try:
        wikipedia.set_lang("fr")
        result = wikipedia.page(sub)
        await ctx.send(result.summary)
    except:
        await ctx.send("Page: "+sub+" Introuvable")


@bot.command()
async def postfixe(ctx, a):
    def estVide(pile):
        return len(pile) == 0

    def depiler(pile):
        pile.pop()
        pile.pop()
        return pile

    def detection(chaine):
        chiffre = 0
        ope = 0
        if chaine[0] == ':' or chaine[0] == '+' or chaine[0] == '*' or chaine[0] == '-':
            return False
        if chaine[1] == ':' or chaine[1] == '+' or chaine[1] == '*' or chaine[1] == '-':
            return False
        for char in chaine:
            if ord(char) >= 48 and ord(char) <= 57:
                chiffre += 1
            if char == '*' or char == '+' or char == ':' or char == '-':
                ope += 1
        return chiffre == ope+1

    if not detection(a):
        await ctx.send("Il y a une erreur dans l'expression postfixée")
        return

    pile = []
    i = 0
    for w in a:
        res = 0
        if w == '-':
            if not estVide(pile):
                res = pile[len(pile)-2]-pile[len(pile)-1]
                pile = depiler(pile)
                pile.append(res)
        if w == '+':
            if not estVide(pile):
                res = pile[len(pile)-2]+pile[len(pile)-1]
                pile = depiler(pile)
                pile.append(res)
        if w == '*':
            if not estVide(pile):
                res = pile[len(pile)-2]*pile[len(pile)-1]
                pile = depiler(pile)
                pile.append(res)
        if w == ':':
            if pile[len(pile)-1] == 0:
                await ctx.send("Erreur dans l'expression: on ne peut diviser par 0")
                return
            if not estVide(pile):
                res = pile[len(pile)-2]/pile[len(pile)-1]
                pile = depiler(pile)
                pile.append(res)
        if ord(w) >= 48 and ord(w) <= 57:
            pile.append(int(w))
        i += 1
    await ctx.send(f"Résultat: {pile[0]}")


@bot.command()
async def team(ctx, players):
    players = players.split(",")
    team1 = []
    team2 = []
    while len(team1) != 5:
        trg = randint(0, 9)
        if players[trg] not in team1:
            team1.append(players[trg])

    while len(team2) != 5:
        trg = randint(0, 9)
        if players[trg] not in team2 and players[trg] not in team1:
            team2.append(players[trg])

    await ctx.send(f"Première équipe :\n  -{team1[0]} \n  -{team1[1]} \n  -{team1[2]} \n  -{team1[3]} \n  -{team1[4]} \n\n Deuxième Équipe :\n  -{team2[0]} \n  -{team2[1]} \n  -{team2[2]} \n  -{team2[3]} \n  -{team2[4]}")


@bot.command()
async def t(ctx, *msg):
    message = " ".join(msg)
    f = open("model.txt", "r")
    if len(f.readlines()) >= 35:
        with open('model.txt', 'w') as f:
            f.write("The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Salut, qui est tu ?\nAI: Je suis une IA et mon prénom est RFK.\n")
    f.close()
    f = open("model.txt", "r")
    training = "\n".join(f.readlines())
    f.close()
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"{training}\n\nHuman: {message}\nAI:",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=["\n", " Human:", " AI:"]
    )
    f = open("model.txt", "a")
    f.write("Human: "+message+"\n")
    f.write("AI:"+response["choices"][0]["text"].split("\n")[0]+"\n")
    f.close()
    await ctx.send(response["choices"][0]["text"].split("\n")[0])


@bot.command()
async def play(ctx, *, query):
    # Récupérer le salon vocal dans lequel se trouve l'utilisateur
    channel = ctx.message.author.voice.channel

    # Rejoindre le salon vocal
    await channel.connect()

    # Rechercher la vidéo en utilisant l'API de recherche YouTube
    # et récupérer l'URL de la première vidéo de la liste de résultats
    video_url = search_video_on_youtube(query)

    # Créer un objet AudioSource pour la vidéo
    source = discord.FFmpegPCMAudio(video_url)

    # Ajouter la source au lecteur de musique
    ctx.voice_client.play(source)
    print(f"Now playing: {query}...")

bot.run(os.environ['DISCORD_TOKEN'])
