import discord
import random
import math
from discord.ext import commands
from discord.ext.commands import context
from bs4 import BeautifulSoup
import aiohttp
import openai
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from time import sleep

openai.api_key = "tkt"
intents = discord.Intents.default()
intents.message_content = True
intents.guild_messages = True
intents.guild_reactions = True
intents.members = True 
client = discord.Client(intents=intents)

bot = commands.Bot(command_prefix="§",description="Beware...", intents=intents)
bot.remove_command('help')

useRLTrack = False

#-----------------FONCTION EXTERNE------------------------#

def lire_fichier(nom_fichier, encodage='utf-8'):
    try:
        with open(nom_fichier, 'r') as fichier:
            contenu = fichier.read()
        return contenu
    except FileNotFoundError:
        return "Le fichier spécifié n'a pas été trouvé."
    except Exception as e:
        return f"Une erreur s'est produite : {str(e)}"
    
def correct(text: str):
    text2 = text.replace('Ã', 'à')
    text2 = text2.replace('à©', 'é')
    text2 = text2.replace('à¨', 'è')
    text2 = text2.replace('àª', 'ê')
    text2 = text2.replace('à¹', 'ù')
    text2 = text2.replace('à§', 'ç')
    return text2
#---------------------------------------------------------#

#------------------------MUSIC----------------------------#


#--------------------------------------------------------#

@bot.event
async def on_ready():
    print("I'm ready for Shacoterie!")
    
@bot.command()
async def bonjour(ctx):
    print("Salut.")
    await ctx.send("Salut.")

@bot.command()
async def help(ctx):
    texte = lire_fichier("help.txt", encodage='utf-8')
    await ctx.send(correct(texte)) 

#Fonction random
@bot.command()
async def rand(ctx, nbr1 : int , nbr2 : int ):
    if(nbr2 > 100000000000000000000000000000000):
        ctx.send("Le nombre entrée est trop long.")
        return
    else:
        if (nbr1 > nbr2):
            nbre = random.randint(nbr2, nbr1)
        else:
            nbre = random.randint(nbr1, nbr2)
        await ctx.send(f"**Résultat:** {nbre}")

@bot.command()
async def ask(ctx, *, question):
    try:
        # Utilisez l'API GPT-3 pour obtenir une réponse à la question
        response = openai.Completion.create(
            engine="davinci",  # Utilisez le moteur GPT-3 de votre choix
            prompt=question,
            max_tokens=100,  # Limitez la longueur de la réponse
            stop=None  # Vous pouvez spécifier des mots pour arrêter la réponse ici
        )

        # Envoyez la réponse au canal Discord
        await ctx.send(response.choices[0].text)
    except Exception as e:
        await ctx.send(f"Une erreur s'est produite : {str(e)}")

# RL Tracker -------------------------------------------------------

def recuperer_page_html(url):
    # Configuration de Selenium pour utiliser Chrome en mode headless
    options = Options()
    options.headless = True

    # Créer une instance du navigateur Chrome
    driver = webdriver.Firefox()
 
    try:
        # Charger la page Web
        driver.get(url)
        sleep(2)
        # Attendre un certain temps pour s'assurer que la page est chargée
        # Vous pouvez ajuster le temps d'attente en fonction de votre cas d'utilisation

        # Récupérer le contenu HTML brut de la page
        page_html = driver.page_source
        
        return page_html
    except Exception as e:
        print("Erreur lors de la récupération de la page HTML :", e)
        return None
    finally:
        driver.quit()  # Fermer le navigateur après utilisation

# Exemple d'utilisation
def get_url(pseudo, platform):
    return f"https://rocketleague.tracker.network/rocket-league/profile/{platform}/{pseudo}/overview"  # Remplacez par l'URL de la page que vous souhaitez récupérer

def getElo_Global(html_content):
    try:
        # Utilisez le parseur lxml pour analyser le HTML
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Recherchez toutes les balises <span> avec les attributs spécifiés
        span_elements = soup.find_all('span', {'data-v-45c45a4a': True, 'class': 'value'})
        
        # Vérifiez si des balises <span> ont été trouvées
        if span_elements:
            # Récupérez les textes à l'intérieur des 7 premières balises <span>
            valeurs = [span.text for span in span_elements[:7]]
            return valeurs
        else:
            print("Les balises <span> spécifiées n'ont pas été trouvées.")
            return None
    except Exception as e:
        print("Erreur lors de l'extraction des numéros :", e)
        return None
    
def getElo_rank(html_content):
    try:
        # Utilisez le parseur lxml pour analyser le HTML
        soup = BeautifulSoup(html_content, 'lxml')

        # Recherchez toutes les balises <span> avec les attributs spécifiés
        span_elements = soup.find_all('div', {'data-v-12ed86bb': True, 'class': 'value'})
        span_elements2 = soup.find_all('div', {'data-v-12ed86bb': True, 'class': 'rank'})
        span_elements3 = soup.find_all('img', {'data-v-12ed86bb': True, 'class': 'icon'})
        span_elements4 = soup.find_all('div', {'data-v-12ed86bb': True, 'class': 'playlist'})
        span_elements5 = soup.find_all('span', {'data-v-45c45a4a': True, 'class': 'value'})
        
        # Vérifiez si des balises <span> ont été trouvées
        if span_elements and span_elements2 and span_elements3 and span_elements4 and span_elements5:
            # Récupérez les textes à l'intérieur des 7 premières balises <span>
            valeurs = [span.text for span in span_elements[:30]]
            valeurs2 = [span.text for span in span_elements2[:30]]
            valeurs3 = [img['src'] for img in span_elements3[:30]]
            valeurs4 = [span.text for span in span_elements4[:30]]
            valeurs5 = [span.text for span in span_elements5[:30]]
            
            return valeurs, valeurs2, valeurs3, valeurs4, valeurs5
        else:
            print("Les balises <span> spécifiées n'ont pas été trouvées.")
            return None
    except Exception as e:
        print("Erreur lors de l'extraction des numéros :", e)
        return None

def findFirst(tab, c):
    for i, chaine in enumerate(tab):
        if c in chaine:
            return i
    return -1

def calculProb(nbr1, nbr2):
    result = (math.pow(nbr1/10,30) / (math.pow(nbr1/10,30) + math.pow(nbr2/10,30)))*100
    return result

def nbrMatch(nom_fichier, pseudo):
    try:
        with open(nom_fichier, 'r') as fichier:
            for ligne in fichier:
                elements = ligne.strip().split()
                if len(elements) >= 2 and elements[0] == pseudo:
                    # Convertir les éléments suivant le pseudo en entiers et les additionner
                    nombres = [int(x) for x in elements[1:]]
                    somme = sum(nombres)
                    return somme
        # Si le pseudo n'est pas trouvé, retourner None ou une valeur par défaut
        return None
    except FileNotFoundError:
        print(f"Le fichier {nom_fichier} n'a pas été trouvé.")
        return None

@bot.command()
async def RLnb1(ctx, pseudo: str):
    somme = nbrMatch("nbrMatch1.txt", pseudo)
    await ctx.send(f"**{pseudo}** a fait **{somme}** 1V1 en totalité.")

@bot.command()
async def RLnb2(ctx, pseudo: str):
    somme = nbrMatch("nbrMatch2.txt", pseudo)
    await ctx.send(f"**{pseudo}** a fait **{somme}** 2V2 en totalité.")
    
@bot.command()
async def RLvs(ctx, platform: str, pseudo: str, platform2, pseudo2):
    url = get_url(pseudo, platform)
    html = recuperer_page_html(url)
    elo, rank, icon, cate, stats = getElo_rank(html)
    url = get_url(pseudo2, platform2)
    html = recuperer_page_html(url)
    elo2, rank2, icon2, cate2, stats2 = getElo_rank(html)
    index = findFirst(cate, "Ranked Duel 1v1 ")
    index2 = findFirst(cate2, "Ranked Duel 1v1 ")

    print(index)
    if len(elo) > index*3+2 and len(elo2) > index2*3+2:
        prob =calculProb(int(elo[index*3].replace(',','')), int(elo2[index*3].replace(',','')))
    elif len(elo) < index*3+2 and len(elo2) < index2*3+2:
        prob =calculProb(int(elo[index*2].replace(',','')), int(elo2[index*2].replace(',','')))
    elif len(elo) > index*3+2 and len(elo2) < index2*3+2:
        prob =calculProb(int(elo[index*3].replace(',','')), int(elo2[index*2].replace(',','')))
    else:
        prob =calculProb(int(elo[index*2].replace(',','')), int(elo2[index*3].replace(',','')))
    await ctx.send(f"La probabilité que {pseudo} gagne contre {pseudo2} en 1VS1 sur Rocket League est de: {prob}%")


@bot.command()
async def RLtracko(ctx, platform: str, pseudo: str):
    url = get_url(pseudo, platform)
    html = recuperer_page_html(url)
    result = ""
    elo, rank, icon, cate, stats = getElo_rank(html)
    if len(elo) > ((len(cate)-1)*3+2):
        for index in range(0, len(cate)):
            result = result + f"**{cate[index]}**\n{rank[index*2]}, **{elo[index*3]}** [{elo[(index*3)+1]}] ({rank[(index*2)+1]}), match:{elo[(index*3)+2]}\n"
    else:
        for index in range(0, len(cate)):
            result = result + f"**{cate[index]}**\n{rank[index*2]}, **{elo[index*2]}** ({rank[(index*2)+1]}), match:{elo[(index*2)+1]}\n"
    result = result + f"**Wins**: {stats[7]} | **Goal Shot Ratio**: {stats[8]}\n**Goals**: {stats[9]} | **Shots**: {stats[10]}\n**Assists**: {stats[11]} | **Saves**: {stats[12]}\n**MVPs**: {stats[13]} | **TRN Score**: {stats[14]}\n"
    await ctx.send(result)

def RLTrackInfo(platform: str, pseudo: str, category: str):
    url = get_url(pseudo, platform)
    html = recuperer_page_html(url)
    elo, rank, icon, cate, stats = getElo_rank(html)
    
    if not elo or not rank or not icon or not cate:
        return -1, -1

    index = findFirst(cate, category)
    print(index)
    
    if index == -1:
        return -1, -1

    # Vérifier que les listes elo et rank ont suffisamment d'éléments
    if len(elo) > index*3+2 and len(rank) > (index*2)+1:
        if (elo[(index*3)+1] == ' '):
            elo[(index*3)+1] = "Nothing"
        result = f"**Information {pseudo} in {category}**\n**Rank:** {rank[index*2]}\n**Ranking:** {rank[(index*2)+1]}\n**Elo:** {elo[index*3]}\n**Peak Elo:** {elo[(index*3)+1]}\n**Number of match:** {elo[(index*3)+2]}"
    else:
        result = f"**Information {pseudo} in {category}**\n**Rank:** {rank[index*2]}\n**Ranking:** {rank[(index*2)+1]}\n**Elo:** {elo[index*2]}\n**Number of match:** {elo[(index*2)+1]}"
    iconRank = icon[index]
    return result, iconRank

@bot.command()
async def RLtrack1(ctx, platform: str, pseudo: str):
    message, icon = RLTrackInfo(platform, pseudo, "Ranked Duel 1v1 ")
    if (message == -1):
        await ctx.send("**Rank Invalide**")
    else:
        await ctx.send(message)
        await ctx.send(icon)

@bot.command()
async def RLtrack2(ctx, platform: str, pseudo: str):
    message, icon = RLTrackInfo(platform, pseudo, "Ranked Doubles 2v2 ")
    if (message == -1):
        await ctx.send("**Rank Invalide**")
    else:
        await ctx.send(message)
        await ctx.send(icon)

@bot.command()
async def RLtrack3(ctx, platform: str, pseudo: str):
    message, icon = RLTrackInfo(platform, pseudo, "Ranked Standard 3v3 ")
    if (message == -1):
        await ctx.send("**Rank Invalide**")
    else:
        await ctx.send(message)
        await ctx.send(icon)

@bot.command()
async def RLtrackh(ctx, platform: str, pseudo: str):
    message, icon = RLTrackInfo(platform, pseudo, "Hoops ")
    if (message == -1):
        await ctx.send("**Rank Invalide**")
    else:
        await ctx.send(message)
        await ctx.send(icon)

@bot.command()
async def RLtrackr(ctx, platform: str, pseudo: str):
    message, icon = RLTrackInfo(platform, pseudo, "Rumble ")
    if (message == -1):
        await ctx.send("**Rank Invalide**")
    else:
        await ctx.send(message)
        await ctx.send(icon)

@bot.command()
async def RLtrackd(ctx, platform: str, pseudo: str):
    message, icon = RLTrackInfo(platform, pseudo, "Dropshot ")
    if (message == -1):
        await ctx.send("**Rank Invalide**")
    else:
        await ctx.send(message)
        await ctx.send(icon)

@bot.command()
async def RLtracks(ctx, platform: str, pseudo: str):
    message, icon = RLTrackInfo(platform, pseudo, "Snowday ")
    if (message == -1):
        await ctx.send("**Rank Invalide**")
    else:
        await ctx.send(message)
        await ctx.send(icon)

@bot.command()
async def RLtrackt(ctx, platform: str, pseudo: str):
    message, icon = RLTrackInfo(platform, pseudo, "Tournament Matches ")
    if (message == -1):
        await ctx.send("**Rank Invalide**")
    else:
        await ctx.send(message)
        await ctx.send(icon)

#----------------------------GPT------------------------------------#


def gpt(texte):
    try:
        # Configuration de Selenium pour utiliser Chrome en mode headless
        options =   Options()
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36")
        options.headless = True

        # Créer une instance du navigateur Chrome
        driver = webdriver.Firefox()

        # Charger la page Web
        driver.get("https://chat.openai.com/c/01427041-994d-4b82-b6e1-2db3f6ad180e")  # Remplacez "URL_DE_VOTRE_PAGE" par l'URL réelle
        sleep(1)
        # Trouver l'élément textarea par son ID
        textarea = driver.find_element_by_id("prompt-textarea")
        sleep(1)
        # Effacer tout texte existant dans la textarea (facultatif)
        textarea.clear()
        sleep(1)
        # Écrire le texte dans la textarea
        textarea.send_keys(texte)

        # Trouver l'élément du bouton par son attribut "data-testid"
        bouton = driver.find_element_by_css_selector('[data-testid="send-button"]')

        # Cliquer sur le bouton
        bouton.click()
        sleep(30)
        # Fermer le navigateur après utilisation
        driver.quit()
    except Exception as e:
        print("Erreur lors de l'envoi du texte et de la simulation de l'appui sur le bouton :", e)

@bot.command()
async def oui(ctx, texte):
    gpt(texte)
    await ctx.send("gg")

bot.run("tkt")