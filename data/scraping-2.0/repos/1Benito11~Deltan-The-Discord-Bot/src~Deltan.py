from __future__ import unicode_literals

import asyncio
import datetime
import json
import os
import random
import re
import sys
import threading
import time
from datetime import datetime, time, timedelta
from typing import Literal, Optional

import discord
import feedparser
import mysql.connector
import openai
import polling
import pytz
import requests
import youtube_dl
from asgiref.sync import sync_to_async
from bs4 import BeautifulSoup
from discord.ext import commands
from discord.ext.commands import Context, Greedy
from discord.utils import get
from geopy import geocoders
from timezonefinder import TimezoneFinder

discord.utils.setup_logging()

my_guild = discord.Object(id=804410750147231814)
jp_guild = discord.Object(id=943102076974661693)

guildz = [my_guild, jp_guild]

# api keys
if os.path.isfile('./Deltan-The-Discord-Bot/apis/openai.txt'):
    with open('./Deltan-The-Discord-Bot/apis/openai.txt') as txt:
        token = txt.readline()
        openai.api_key = token
else:
    openai.api_key = os.environ['open_ai']


if os.path.isfile('./Deltan-The-Discord-Bot/apis/wapi.txt'):
    with open('./Deltan-The-Discord-Bot/apis/wapi.txt') as txt:
        token = txt.readline()
        wapii = token
else:
    wapii = os.environ['wapi']

intents = discord.Intents.default()
intents.members = True
intents.message_content = True

if os.path.isfile('./Deltan-The-Discord-Bot/apis/db.txt'):
    with open('./Deltan-The-Discord-Bot/apis/db.txt') as txt:
        listt = txt.readlines()
        user = str(listt[0]).strip()
        password = str(listt[1]).strip()
        host = str(listt[2]).strip()
else:
    user = os.getenv("user").strip()
    password = os.getenv("password").strip()
    host = os.getenv("host").strip()

# baza danych
mydb = mysql.connector.connect(
    user=user,
    password=password,
    host=host,
    database='sofwv9c48eandcct',
    port='3306'
)

cursor = mydb.cursor()

if (mydb.is_connected()):
    print('database connected\n')
else:
    print('database connection failed\n')


# prefix
global prefixx


def prefix(client, message):
    cursor.execute("SELECT server FROM prefix")

    tupel = cursor.fetchall()
    results = [i[0] for i in tupel]

    for x in results:
        if str(message.guild.id) == x:
            cursor.execute(f'SELECT prefix FROM prefix WHERE server="{x}"')
            pre_results = cursor.fetchone()
            prefixx = str(pre_results[0])
            break
        else:
            prefixx = '!'

    return prefixx


client = commands.Bot(command_prefix=prefix, help_command=None, intents=intents,
                      allowed_mentions=discord.AllowedMentions(everyone=True))

client.tree.copy_global_to(guild=my_guild)


@client.event
async def on_ready():
    channel = client.get_channel(805079580225699860)
    await channel.send('im ready to fight')
    print('Deltan is ready')
    sync = await client.tree.sync()
    print(f'synced {len(sync)} commands')
    await client.change_presence(status=discord.Status.idle, activity=discord.Game('Δ + n'))
    background_zast()


def background_zast():
    zast_thread = threading.Thread(target=api_check)
    zast_thread.start()


@client.event
async def on_member_join(member):
    # programista specjalny
    default_role = get(member.guild.roles, id='943103438340915200')
    print(default_role)
    await client.add_roles(member, default_role)


@client.event
async def on_message(message):
    channel = message.channel
    if message.author.id == 487954704731734037:
        return
    if channel.id == 1060704935060189285:
        await chatbot(message)
    if channel.id == 1061085341118902334:
        await chatbot(message)

    if any(x in message.content for x in ['polska', 'polske', 'polski', 'polskie']):
        await channel.send(file=discord.File("Deltan-The-Discord-Bot/assets/videos/jp.mp4"), reference=message)
    elif message.content == 'piwo':
        await channel.send('to moje paliwo')
    elif message.content == 'paliwo':
        await channel.send('to moje piwo')
    elif message.author.id == 676865845217198083:
        emotki = [
            '<:toja:1041131016095539311>',
            '<:czarny:943115717828571136>',
            '<:tatozajebiscie:945419314121572383>',
            '<:adas:945411599940857856>',
            '<:totylkoczarny:945419871498403891>',
            '<:adamek:945420069725413466>',
            '<:THICCADAM:945423073551728711>',
        ]
        for emoji in emotki:
            await message.add_reaction(emoji)
    elif message.author == client.user:
        return
    elif message.content == 'sus':
        await channel.send('sussy baka amogus')
    elif message.content == 'o/':
        await channel.send('o/')
    elif message.content == '\o':
        await channel.send('\o')
    if message.content == 'kk':
        await channel.send('ja kiedy ok')
    if message.content.startswith('ben') or message.content.startswith('Ben'):
        liczba = random.randrange(1000)
        chek = random.randrange(1000)
        if liczba != chek:
            Responses = [' Yes https://c.tenor.com/Pta1QQlnZZYAAAAC/ben-yes-yes.gif',
                        " No. https://c.tenor.com/SdsYv4vylh0AAAAC/dog-saying-no-no.gif",
                        " *Odkłada telefon* https://c.tenor.com/VfB8CeuNh-0AAAAM/dog-hang-up-the-call-ben-hang-up.gif",
                        " HO HO HO” https://c.tenor.com/01tnAz3pRFwAAAAM/ben-laughs-ben-laughing.gif",
                        " Blee https://image.winudf.com/v2/image/Y29tLm91dGZpdDcudGFsa2luZ2Jlbl9zY3JlZW5fMTNfMTUzMDYxMDE5MV8wMjE/screen-5.jpg?fakeurl=1&type=.jpg"
                        ]
            await channel.send(f'{random.choice(Responses)}')
        else: 
            await channel.send(f'https://tenor.com/view/izak-tubson-jasper-tubson_-izakzawiedzenie-gif-25635309')
    if message.content == '!resetprefix':
        if message.author.top_role.permissions.administrator == True:
            print(message.guild.id)
            cursor.execute(
                f'UPDATE prefix SET prefix = "!" WHERE server = {message.guild.id}')
            mydb.commit()
            await channel.send('Prefix zresetowany do !')
        else:
            await channel.send('nie masz uprawnień lol')
    if client.user.mention in message.content.split():
        cursor.execute(
            f'SELECT prefix FROM prefix WHERE server="{message.guild.id}"')
        pre_results = cursor.fetchone()
        prefixz = str(pre_results[0])
        await channel.send(f'Mój prefix w tym serwerze to:  {prefixz}  (jak jesteś adminem to możesz zrestować za pomocą !resetprefix)')

    await client.process_commands(message)


@commands.has_permissions(administrator=True)
@client.hybrid_command(name='admin', description='jeśli to widzisz to masz admina, gratulacje (chyba, że nie działa xd)')
async def admin(ctx: commands.Context):
    await ctx.send('ok')


@client.hybrid_command(name='ping', description='Pokazuje ping bota')
async def ping(ctx: commands.Context):
    await ctx.send(f'Mój ping to: {round(client.latency * 1000)}ms ')


@client.command()
async def test(ctx):
    print('halooooo')
    await ctx.send('no dziala')
    messag = ctx.fetch_message(id)
    await messag.add_reaction('🗿')

# RNG


@client.hybrid_command(name='rzut_moneta', description='rzut monetą', aliases=['flip', 'moneta', 'coinflip'])
async def coin(ctx: commands.Context):
    coins = ["heads", "tails"]
    await ctx.send(random.choice(coins))


@client.hybrid_command(name='1na10', description='Wysyła losobą liczbę od 1 do 10', aliases=['1z10', '1-10'])
async def _110(ctx: commands.Context):
    rgn = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    await ctx.send(random.choice(rgn))


@client.hybrid_command(name='kość', description='Rzuca kością', aliases=['kostka', 'dice'])
async def dice(ctx: commands.Context):
    dices = ['1', '2', '3', '4', '5', '6']
    await ctx.send(random.choice(dices))


# różne
@client.hybrid_command(name='ok', description='ja kiedy ok ok kk')
async def ok(ctx: commands.Context):
    await ctx.send('ok')


@client.hybrid_command()
async def dm(ctx: commands.Context):
    user = ctx.author
    await user.send("is this working?")


@client.hybrid_command(name='pies', description='hau hau', aliases=['hau', 'woof'])
async def bark(ctx: commands.Context):
    await ctx.send('hau hau')


@client.command()
async def tf(ctx):
    await ctx.send('https://i.imgur.com/jWr67J8.png')


@client.hybrid_command(name="mama", description='Sprawdź')
async def mom(ctx):
    user = ctx.author
    print(f'{user} has fallen for my trap XD')
    await user.send('twoja mama')


@client.hybrid_command(name='shrek', description='Wysyła shreka... tak, w całości')
async def shrek(ctx: commands.Context) -> None:
    await ctx.send('https://i.imgur.com/IsWDJWa.mp4')


# inside żarty

@client.command()
async def wacek(ctx):
    await ctx.send('<@!487954704731734037>')

@client.hybrid_command(name='adam', description='Szkaluje Adama', aliases=['nword', 'czarny'])
async def adam(ctx: commands.Context):
    responses = [' Śmierdzielu',
                 " Śmierdzisz",
                 " Lamusie",
                 " Gałganie",
                 " Wyglądasz jak z Czarnobylu",
                 " Ty deklu",
                 " Menelu",
                 " Umyj sie",
                 " Brudasie",
                 " Leniu",
                 " Mieszkasz w kuchni",
                 " Ty robaku",
                 " Ty małpo",
                 " Ty huncwacie",
                 " Twoje zęby są jak gwiazdy - wielkie, żółte i daleko od siebie.",
                 " Ty flądro niemyta",
                 " Lecz sie na nogi bo na glowe juz za pozno",
                 " Ta szpara miedzy zebami to na zetony",
                 " twoja mama",
                 " 亚当，把自己洗脏"
                 ]
    emotki = ['<:czarny:943115717828571136>',
              '<:tatozajebiscie:945419314121572383>',
              '<:adas:945411599940857856>',
              '<:totylkoczarny:945419871498403891>',
              '<:adamek:945420069725413466>',
              '<:THICCADAM:945423073551728711>',
              ]
    await ctx.send(f' {random.choice(emotki)} {random.choice(responses)} <@!676865845217198083>')


@client.command()
async def gama(ctx):
    await ctx.send('Dobra gama, mainujesz Yasuo <@!595589296145039372>')


@client.command()
async def bartus(ctx):
    await ctx.send('Beka z ciebie barti, mainujesz szczura teemo-ka <@!552560586119053323>')


@client.command()
async def yuumi(ctx):
    await ctx.send('dobra szczylu zachfatwany, mainujesz yuumi <@!466597814693003277>')

# emotes


@client.command()
async def pepega(ctx):
    await ctx.send('https://imgur.com/gallery/Ir8VKVz')


@client.command()
async def poggers(ctx):
    await ctx.send('https://imgur.com/gallery/GTukTtS')


@client.command()
async def monkas(ctx):
    await ctx.send('https://imgur.com/gallery/uf9YNVm')


@client.command()
async def ez(ctx):
    await ctx.send('https://imgur.com/gallery/H73d7ji')


# bardziej skomplikowane komendy


@client.hybrid_command(name='time', description='Pokazuje czas w danym miejscu na świecie', aliases=['t',])
async def time(ctx: commands.Context, *, city: str):
    g = geocoders.GeoNames(username="benito12")
    l = g.geocode(city)
    lat = l.latitude
    lng = l.longitude
    api_key = wapii
    url = "https://api.openweathermap.org/data/2.5/onecall?lat=%s&lon=%s&appid=%s&units=metric" % (
        lat, lng, api_key)
    tf = TimezoneFinder()
    latitude = lat
    longitude = lng
    timezone = tf.timezone_at(lng=longitude, lat=latitude)
    timezone1 = pytz.timezone(timezone)
    now = datetime.now(timezone1)
    local = (now.strftime("%H:%M:%S"))
    embed = discord.Embed(title="Time", color=0xff4f4f)
    embed.add_field(name='Time', value='In' +
                    {city}+{timezone} + 'it is:'+{local}, inline=True)
    await ctx.send(embed=embed)


@time.error
async def info_error(ctx, error):
    if isinstance(error, commands.CommandError):
        await ctx.send("This city doesn't exist or you spelled it wrong, remember to write spaces as _ and capitalization (example: Los_Angeles). And remember that some cities can have same name so it might show wrong timezone")


@client.hybrid_command(name='weather', description='Pokazuje prognoze pogody w danym mieście', aliases=['w', 'weat'])
async def weather(ctx: commands.Context, city: str):
    g = geocoders.GeoNames(username="benito12")
    l = g.geocode(city)

    lat = l.latitude
    lon = l.longitude
    api_key = wapii

    url = "https://api.openweathermap.org/data/2.5/onecall?lat=%s&lon=%s&appid=%s&units=metric" % (
        lat, lon, api_key)
    response = requests.get(url)
    data = json.loads(response.text)

    temp = data["current"]["temp"]
    pres = data["current"]["pressure"]
    wilgotnosc = data["current"]["humidity"]
    clouds = data["current"]["clouds"]
    wind = data["current"]["wind_speed"]
    main = data["current"]['weather'][0]['main']
    desc = data["current"]["weather"][0]['description']
    visit = data["current"]['visibility']
    print(clouds)

    cloud = 'https://i.imgur.com/JF9QU7A.png'
    cloudcolor = 0x50504c

    cloudsun = 'https://i.imgur.com/8SoDHhW.png'
    cloudsuncolor = 0xd9dceb

    suncloud = 'https://i.imgur.com/cZPk3Qy.png'
    suncloudcolor = 0xf1eea0

    sun = 'https://i.imgur.com/61hWpgK.png'
    suncolor = 0xfff700

    if clouds >= 1 and clouds <= 25:
        print('work')
        embed = discord.Embed(title="Weather", color=suncolor)

        embed.set_author(name="Deltan", icon_url=sun)

        embed.add_field(
            name="Weather", value=f"Current Forecast:  {main}, {desc}", inline=False)
        embed.add_field(name="Temperature",
                        value=f"Current: {temp}°C", inline=False)
        embed.add_field(name="Pressure",
                        value=f"Current: {pres}Pa", inline=False)
        embed.add_field(
            name="Clouds", value=f"Percantage of clouds in the sky: {clouds}%", inline=False)
        embed.add_field(
            name="Humidity", value=f"Current humidity is: {wilgotnosc}%", inline=False)
        embed.add_field(
            name="Wind", value=f"Current wind speed is: {wind} km/h", inline=False)
        embed.add_field(
            name="Visibility", value=f"Current Visibility is: {visit} mi", inline=False)

        await ctx.send(embed=embed)
    else:
        if clouds >= 26 and clouds <= 50:
            print('work')
            embed = discord.Embed(title="Weather", color=suncloudcolor)

            embed.set_author(name="Deltan", icon_url=suncloud)

            embed.add_field(
                name="Weather", value=f"Current Forecast:  {main}, {desc}", inline=False)
            embed.add_field(name="Temperature",
                            value=f"Current: {temp}°C", inline=False)
            embed.add_field(name="Pressure",
                            value=f"Current: {pres}Pa", inline=False)
            embed.add_field(
                name="Clouds", value=f"Percantage of clouds in the sky: {clouds}%", inline=False)
            embed.add_field(
                name="Humidity", value=f"Current humidity is: {wilgotnosc}%", inline=False)
            embed.add_field(
                name="Wind", value=f"Current wind speed is: {wind} km/h", inline=False)
            embed.add_field(
                name="Visibility", value=f"Current Visibility is: {visit} mi", inline=False)

            await ctx.send(embed=embed)
        else:
            if clouds >= 51 and clouds <= 75:
                print('work')
                embed = discord.Embed(title="Weather", color=cloudsuncolor)

                embed.set_author(name="Deltan", icon_url=cloudsun)

                embed.set_thumbnail(url=cloudsun)
                embed.add_field(
                    name="Weather", value=f"Current Forecast:  {main}, {desc}", inline=False)
                embed.add_field(name="Temperature",
                                value=f"Current: {temp}°C", inline=False)
                embed.add_field(name="Pressure",
                                value=f"Current: {pres}Pa", inline=False)
                embed.add_field(
                    name="Clouds", value=f"Percantage of clouds in the sky: {clouds}%", inline=False)
                embed.add_field(
                    name="Humidity", value=f"Current humidity is: {wilgotnosc}%", inline=False)
                embed.add_field(
                    name="Wind", value=f"Current wind speed is: {wind} km/h", inline=False)
                embed.add_field(
                    name="Visibility", value=f"Current Visibility is: {visit} mi", inline=False)
                await ctx.send(embed=embed)

            else:
                if clouds >= 76 and clouds <= 100:
                    print('work')
                    embed = discord.Embed(
                        title=f"Current Weather in {city}", color=cloudcolor)

                    embed.set_author(name="Deltan", icon_url=cloud)

                    embed.add_field(
                        name="Weather", value=f"Current Forecast:  {main}, {desc}", inline=False)
                    embed.add_field(name="Temperature",
                                    value=f"Current: {temp}°C", inline=False)
                    embed.add_field(name="Pressure",
                                    value=f"Current: {pres}Pa", inline=False)
                    embed.add_field(
                        name="Clouds", value=f"Percantage of clouds in the sky: {clouds}%", inline=False)
                    embed.add_field(
                        name="Humidity", value=f"Current humidity is: {wilgotnosc}%", inline=False)
                    embed.add_field(
                        name="Wind", value=f"Current wind speed is: {wind} km/h", inline=False)
                    embed.add_field(
                        name="Visibility", value=f"Current Visibility is: {visit} mi", inline=False)
                    await ctx.send(embed=embed)
                else:
                    await ctx.send('Somethin didnt worked out')


# @client.hybrid_command(aliases=['tic', 'tac', 'toe', 'ttt'])
# async def tic_tac_toe(ctx, user1):
#     user = ctx.author
#     channel = ctx.channel
#     board = f' {0},{1},{2}\n {3},{4},{5}\n {6},{7},{8}\n'
#     wincond = [
#         [0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8],
#         [0, 3, 6],
#         [1, 4, 7],
#         [2, 5, 8],
#         [0, 4, 8],
#         [2, 4, 6]


#     ]

#     await ctx.send(f"{board}\n It's {user}'s turn")

#     msg = await client.wait_for("message", check=check)
#     if msg.author != user and msg.channel != channel and message.content != ['1', '2', '3', '4', '5', '6', '7', '8', '0']:
#         return


# embed messages


@client.hybrid_command(name='info', description='Pokazuje info o bocie')
async def info(ctx: commands.Context):
    embed = discord.Embed(title="Info", url="https://github.com/1Benito11/Deltan-The-Discord-Bot",
                          description="Bot, co istnieje", color=0x38bdff)
    embed.set_author(
        name="Deltan", url="https://github.com/1Benito11/Deltan-The-Discord-Bot")
    embed.set_thumbnail(url="https://i.imgur.com/OGzXcQm.png")
    embed.add_field(
        name="Po co", value="Bot został zrobiony do dwóch rzeczy: nauki pythonga i szkalowania adama.", inline=True)
    embed.add_field(
        name="Jak", value="Python 3.10 i discord.py 2.1.0", inline=True)
    embed.add_field(name="Komendy", value="!help", inline=True)
    embed.add_field(
        name="Więcej", value="Więcej informacji na https://github.com/1Benito11/Deltan-The-Discord-Bot", inline=True)
    embed.set_footer(text="Δ+n")
    await ctx.send(embed=embed)


# @client.hybrid_command()
# async def help(ctx):
#     embed = discord.Embed(
#         title="Help", description="Info about commands, WIP", color=0xff4f4f)
#     embed.add_field(name="Music", value="!join - join vc,                                                               !play (link) - joins vc that you're in and plays provided link,                                    !pause - pauses music, !resume - resumes music,                           !skip - skips song and moves to next one in queue,                                                     !queue - shows current queue,                                                   !currentsong or !np - shows what song is playing,                                                 !stop - stops entirely playing songs", inline=False)
#     embed.add_field(
#         name="!play (link) - joins vc that you're in and plays provided link", value='{}', inline=False)
#     embed.add_field(name="!pause - pauses music", value='{}', inline=True)
#     embed.add_field(name="!resume - resumes music", value='{}', inline=True)
#     embed.add_field(name='RNG commands',
#                     value='!dice - shows number from 1 to 6, !1-10 - shows random number from 1 to 10')

#     await ctx.send(embed=embed)


# kalkulator i inne sprawy
@client.hybrid_command(name='kalkulator', description='Kalkuluje, przykład: + 20 20; / 150 25', aliases=['calc'])
async def calculate(ctx: commands.Context, operation: str, *, nums: int):
    if operation not in ['+', '-', '*', '/']:
        await ctx.send('Format for this command is - "operation symbol number1 number2" example - "+ 20 20"')
    var = (f' {operation} '.join(nums))
    await ctx.send(f'{var} = {eval(var)}')


# web scraping
# newsy ze strony szkoly
@client.hybrid_command(name='news', description='Pokazuje ogłoszenia ze strony szkoły (jp)', aliases=['strona', 'ogłoszenia'])
async def strona(ctx: commands.Context):

    URL = "https://zseis.zgora.pl/"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    results = soup.find(id="cont_section")
    tytuly = results.find_all("div", class_="news_title")
    artykuly = results.find_all("div", class_="news_content")

    for tytul, artykul in zip(tytuly, artykuly):
        if len(str(artykul)) > 1950:

            firstpart, secondpart = str(artykul.text)[:len(
                str(artykul.text))//2], str(artykul.text)[len(str(artykul.text))//2:]

            htmlstr1, htmlstr2 = firstpart, secondpart
            first = BeautifulSoup(htmlstr1, 'html.parser')
            second = BeautifulSoup(htmlstr2, 'html.parser')

            await ctx.send("_______")
            await ctx.send(tytul.text)
            await ctx.send(first.text)
            await ctx.send(second.text)
            continue

        await ctx.send("_______")
        await ctx.send(tytul.text)
        await ctx.send(artykul.text)


# zastepstwa
@client.hybrid_command(name='zastępstwa', description='Pokazuje zastępstwa', aliases=["zast"])
async def zastepstwa(ctx: commands.Context):
    await ctx.defer()
    channel = ctx.channel
    await zast_send(channel)

# zastepstwa automatyczne


def api_check():
    channel = client.get_channel(1060906028616663060)
    zast_api = 'https://api.elektronplus.pl/subLessons'
    api = feedparser.parse(zast_api)

    polling.poll(
        lambda: feedparser.parse(zast_api, etag=api.etag).status == 200,
        step=600,
        poll_forever=True)

    asyncio.run(zast_send(channel))


async def zast_send(channel):
    if channel == 1060906028616663060:
        await channel.purge(limit=10)
    zast_api = 'https://api.elektronplus.pl/subLessons'
    zast_re = requests.get(zast_api)
    zast = zast_re.json()

    zast_text = str(zast['todaySubLessonsDay'] + '\n' + zast['todaySubLessons'] + '\n\n' + zast['nextDaySubLessonsDay'] + '\n' + zast['nextDaySubLessons']).replace('Kl. 2Td', '**!!! Kl. 2Td !!!**').replace(
        'Jakubowska M.', 'Jakubowska M. (szmt)').replace('Jersz A.', 'Jersz A. (golem XD)').replace('Sapkowska M.', 'Sapkowska M. (szmt)').replace('Piątkowska A.', 'Piątkowska A. (szmt)')

    zast_tab = []
    total = 0
    nl_counter = 0

    for word in zast_text.splitlines(True):
        if word == '\n':
            nl_counter += 1
            if nl_counter == 2:
                nl_counter = 0
                zast_tab.pop()
                continue
        if total > 1750:
            await channel.send(str(zast_tab).replace(',', '').replace('[', '').replace(']', '').replace("'", '').replace(r'\n', '\n'))
            zast_tab.clear()
            total = 0
            total += len(word)
            zast_tab.append(word)
        else:
            total += len(word)
            zast_tab.append(word)

    if total > 0:
        await channel.send(str(zast_tab).replace(',', '').replace('[', '').replace(']', '').replace("'", '').replace(r'\n', '\n'))

    # if 'Kl. 2Td' in zast_text:
    #     await channel.send('@here')

    if channel == client.get_channel(1060906028616663060):
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        background_zast()


# busy
@client.hybrid_command(name='mzk', description='Pokazuje kiedy busy odjeżdzają z danego przystanku')
async def mzk(ctx: commands.Context, przystanek: str):
    await ctx.defer()
    przystanekk = przystanek.title()

    przystanek_api = 'https://poland-public-transport.konhi.workers.dev/v1/zielonagora/mzk/stops'
    przystanki = requests.get(przystanek_api)
    przystan = przystanki.json()

    FMT = '%H:%M'
    date = datetime.now()
    czas = date.strftime(FMT)

    for x in range(len(przystan)):
        if przystanekk in przystan[str(x)]['name']:
            stop = przystan[str(x)]['id']
            break

    dep_api = f'https://poland-public-transport.konhi.workers.dev/v1/zielonagora/mzk/stops/{stop}/departures'
    dep = requests.get(dep_api)
    departs = dep.json()
    odjazdy = []

    for i in range(len(departs)):
        for y, p in departs[str(i)].items():
            if any(str.isdigit(c) for c in p) == True and 'min' in p:
                res = [int(i) for i in p.split() if i.isdigit()]
                p = res
                y = 'Odjazd za'
                godzina = date + timedelta(minutes=res[0])
                yp = str(str('Godzina odjazdu: ') + str(godzina.time())
                         [:-10] + 'spacekeja' + str(y) + ':' + str(p) + 'min')
                odjazdy.append(yp)
                continue
            elif y == 'time':
                time = departs[str(i)]['time']
                if czas == time:
                    p = '0 min'
                    yp = str(str(y) + ':' + str(czas) +
                             'spacekeja' + 'Odjazd za: ' + str(p))
                else:
                    dep_time = datetime.strptime(
                        time, FMT) - datetime.strptime(czas, FMT)
                    if 'day' in str(dep_time):
                        yp = str(str(y) + ': ' + str(p) + 'spacekeja' +
                                 'Odjazd za:' + str(dep_time).replace(',', ''))
                    else:
                        yp = str(str(y) + ': ' + str(p) +
                                 'spacekeja' + 'Odjazd za: ' + str(dep_time))

                odjazdy.append(yp)

                continue
            else:
                yp = y + p
                odjazdy.append(yp)
                continue

    departs_text = str(odjazdy)
    await ctx.send(departs_text.replace('{', '').replace('}', '').replace(',', '\n').replace("'", '').replace('spacekeja', '\n').replace('[', '').replace(']', '').replace('', '').replace('time', '\nGodzina odjazdu').replace(' destination', 'Kierunek: ').replace(' line', 'Linia: ').replace(' stop', 'Przystanek: ').replace('-1 day', ''))


# @mzk.error
# async def on_command_error(error, ctx):
#     print('asdasd')
#     if isinstance(error, UnboundLocalError):
#         print('error')
#         await ctx.send('Nie znaleziono takiego przystanka, sprawdź pisownię')


# pobieranie yt
@client.hybrid_command(name='yt', description='Pobiera link i wysyła plik w podanym formacie // jak na razie tylko audio działa', aliases=['youtube'])
async def _yt(ctx: commands.Context, link: str, format: str):

    if format == 'audio':
        ydl_opts = {
            'format': 'bestaudio/best',
            'keepvideo': False,
            'audio-quality': '0',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
            'postprocessor_args': [
                '-ar', '16000'
            ],
            'prefer_ffmpeg': True,
            'keepvideo': False,
            'audio-format': 'mp3',
            'output': '%(yt)s.%(mp3)s'
        }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(link, download=False)
        name = info.get('title')
        ydl.download([link])
        for file in os.listdir():
            if file.startswith(name) and file.endswith('.mp3'):
                os.rename(file, 'yt.mp3')

    await ctx.send(file=discord.File("yt.mp3"),)
    os.remove('yt.mp3')


# OPENAI
@client.hybrid_command(name='ai', description='Pogadaj se z AI', aliases=['si', 'openai'])
async def ai(ctx: commands.Context, *, prompt: str):
    await ctx.defer()
    response = await sync_to_async(openai.Completion.create)(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=1980,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    if 'choices' in response:
        if len(response['choices']) > 0:
            await ctx.send('zapytanie: **'+prompt+'**' + response['choices'][0]['text'])
        else:
            await ctx.send("Nie działa, spróbuj ponownie")
    else:
        await ctx.send("Nie działa, spróbuj ponownie")

global history_ai, ai_bot
history_ai = []


@client.hybrid_command(name='chatbot_prompt', description='Zmień defaultowe dane chatbota')
async def chatbot(ctx: commands.Context, *, prompt: str):
    server = str(ctx.guild.id)
    user = str(ctx.author)

    cursor.execute("SELECT server FROM chatbot")

    results = cursor.fetchall()
    tupel = [i[0] for i in results]

    if server in tupel:
        for x in tupel:
            if server == x:
                cursor.execute(
                    f'UPDATE chatbot SET chatbot = "{prompt}", user = "{str(user)}" WHERE server = {x}')
    else:
        cursor.execute(
            f'INSERT INTO chatbot (chatbot, server, user) VALUES ("{prompt}","{str(ctx.guild.id)}","{str(ctx.author)}")')

    mydb.commit()
    await ctx.send(f'Defaultowe dane chatbota zmienione na: {prompt}')
    history_ai = []


def aibot(client, message):
    cursor.execute("SELECT server FROM chatbot")

    tupel = cursor.fetchall()
    results = [i[0] for i in tupel]

    for x in results:
        if str(message.guild.id) == x:
            cursor.execute(f'SELECT chatbot FROM chatbot WHERE server="{x}"')
            pre_results = cursor.fetchone()
            ai_bot = str(pre_results[0])
            break
        else:
            ai_bot = f"To jest konwersacja człowieka z chatbotem AI. Chatbot jest kreatywny, śmieszny i przyjazny.\n\nHuman:{history_ai}"
    history_ai.clear()
    return ai_bot


async def chatbot(message):
    channel = message.channel
    # if message.content == 'reset':
    #     history_ai = []
    #     await channel.send('Historia zresetowana')
    # else:
    if message.content.startswith('!'):
        return
    else:
        if message.author.id == client.user.id:
            ai = 'AI:' + message.content
            history_ai.append(ai)
            print(f'od ai: {history_ai}')
        if message.author.id != client.user.id:
            human = 'Human:' + message.content
            history_ai.append(human)
            print(f'od human: {history_ai}')
            prompt = aibot(client, message) + str(history_ai)
            print(f'prompt: {prompt}')
            response = await sync_to_async(openai.Completion.create)(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0.9,
                max_tokens=1980,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.1,
            )

            if len(history_ai) > 8:
                history_ai.pop(0)
                history_ai.pop(0)

            if 'choices' in response:
                if len(response['choices']) > 0:
                    await channel.send(str(response['choices'][0]['text']).replace('AI:', ''))
                else:
                    await channel.send("Nie działa, spróbuj ponownie")
            else:
                await channel.send("Nie działa, spróbuj ponownie")


@client.hybrid_command(name='testin')
async def testin(ctx):
    channel = ctx.channel
    channel.purge(limit=10)


@client.hybrid_command(name='dalle', description='Robi obrazek na podstawie tego co napisałes', aliases=['dal', 'dall-e', 'dall'])
async def dalle(ctx: commands.Context, *, prompt):
    await ctx.defer()
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    await ctx.send(image_url)


# moderacja ig
@commands.has_permissions(administrator=True)
@client.command()
async def prefix(ctx, prefix: str):
    server = str(ctx.guild.id)
    user = str(ctx.author)

    cursor.execute("SELECT server FROM prefix")

    results = cursor.fetchall()
    tupel = [i[0] for i in results]

    if server in tupel:
        for x in tupel:
            if server == x:
                cursor.execute(
                    f'UPDATE prefix SET prefix = "{prefix}", user = "{str(user)}" WHERE server = {x}')
    else:
        cursor.execute(
            f'INSERT INTO prefix (prefix, server, user) VALUES ("{prefix}","{str(ctx.guild.id)}","{str(ctx.author)}")')

    mydb.commit()
    await ctx.send(f'Prefix zmieniony na: {prefix}')


@commands.has_permissions(administrator=True)
@client.command(pass_context=True, aliases=['clean', 'purge', 'finish'])
async def purge_messages(ctx, amount_to_delete: int = 0):
    if amount_to_delete < 1 or amount_to_delete > 100:
        await ctx.send('Nieprawidłowa ilość')
    else:
        await ctx.channel.purge(limit=amount_to_delete + 1)


# dev tools
@client.command()
async def stopwork(ctx):

    if ctx.author.id == 630157541250170891:
        await ctx.send('cya')
        await client.close()
    else:
        await ctx.send('Nope m8')


@client.command()
async def restart(ctx):

    if ctx.author.id == 630157541250170891:
        await ctx.send('cya')
        os.execv(sys.executable, ['python'] + sys.argv)
    else:
        await ctx.send('Nope m8')


@client.command()
async def status(ctx, status: str, *, line: str):
    print(status, *line)
    if ctx.author.id == 630157541250170891:
        await client.change_presence(status=discord.Status(f'{status}'),  activity=discord.Game(f'{line}'))
    else:
        await ctx.send('XD, chciałbyś')


@client.command()
async def statusres(ctx):
    await client.change_presence(status=discord.Status.idle, activity=discord.Game('Δ + n'))


@client.command()
async def sync(ctx: Context, guilds: Greedy[discord.Object], spec: Optional[Literal["~", "*", "^"]] = None) -> None:
    if not guilds:
        if spec == "~":
            synced = await ctx.bot.tree.sync(guild=ctx.guild)
        elif spec == "*":
            ctx.bot.tree.copy_global_to(guild=ctx.guild)
            synced = await ctx.bot.tree.sync(guild=ctx.guild)
        elif spec == "^":
            ctx.bot.tree.clear_commands(guild=ctx.guild)
            await ctx.bot.tree.sync(guild=ctx.guild)
            synced = []
        else:
            synced = await ctx.bot.tree.sync()

        await ctx.send(
            f"Synced {len(synced)} commands {'globally' if spec is None else 'to the current guild.'}"
        )
        return

    ret = 0
    for guild in guilds:
        try:
            await ctx.bot.tree.sync(guild=guild)
        except discord.HTTPException:
            pass
        else:
            ret += 1

    await ctx.send(f"Synced the tree to {ret}/{len(guilds)}.")


# cogs

@client.command()
async def load(ctx, cog):
    for filename in os.listdir("./src/cogs"):
        if filename.endswith(".py"):
            # cut off the .py from the file name
            await client.load_extension(f"cogs.{filename[:-3]}")
    await ctx.send('cog loaded!')


@client.command()
async def unload(ctx, extension):
    await client.unload_extension(f'cogs.{extension}')
    await ctx.send('cog unloaded!')


# with open('config.txt') as f:
    # TOKEN = f.readline()


@client.hybrid_command(name='github', description='Wysyła link do repo tego bota')
async def github(ctx: commands.Context):
    await ctx.send('Mój github to: https://github.com/1Benito11/Deltan-The-Discord-Bot')


async def load_extensions():
    if os.path.isdir('./Deltan-The-Discord-Bot/src/cogs'):
        for cog in os.listdir('./Deltan-The-Discord-Bot/src/cogs'):
            if cog.endswith('.py'):
                await client.load_extension(f"cogs.{cog[:-3]}")
                print(f'loaded {cog} \n')
    else:
        print(os.listdir)
        for cog in os.listdir():
            if cog.endswith('src/cogs/.py'):
                await client.load_extension(f"cogs.{cog[:-3]}")
                print(f'loaded {cog}')


async def main():
    async with client:
        await load_extensions()

        my_token = os.getenv("token")

        if os.path.isfile('./Deltan-The-Discord-Bot/token.txt'):
            with open('./Deltan-The-Discord-Bot/token.txt') as txt:
                token = txt.readline()
                await client.start(token)
        else:

            await client.start(my_token)

asyncio.run(main())
