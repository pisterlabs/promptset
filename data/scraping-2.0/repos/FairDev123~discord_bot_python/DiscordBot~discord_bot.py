
# [MODULES IMPORT]
import discord
from PIL import Image, ImageFont, ImageDraw
from discord.ext import commands, tasks
from discord.ext.commands import has_permissions, guild_only
from discord.utils import get
from discord import app_commands
from discord.app_commands import Choice
from datetime import datetime
import sqlite3
import random
import math
import asyncio
import time as tm
import requests
import json
import os
import pgapi as pgn
from itertools import cycle
from ps3d_api import PS3D
from io import BytesIO
import openai

with open("./bot_database/clan.json", "r") as t:
    clan_1 = json.load(t)
    hours = clan_1["income_hours"]
    hour1 = hours[0]
    hour2 = hours[1]
    hour3 = hours[2]
    hour4 = hours[3]
    hour5 = hours[4]
    hour6 = hours[5]

# [DATABASE BASE]

con = sqlite3.connect("users.db")
cur = con.cursor()

try:
  create_table = "CREATE TABLE users(id, user, coins, gems)"
  cur.execute(create_table)
  print("Utworzono tablice 'users'")
except:
  pass

# [IMPORTANT THINGS]
walters = ["https://imgur.com/c2dqZkc","https://imgur.com/ZZX6plM","https://imgur.com/IFZmwI2","https://imgur.com/skZPRSE"]

diary = """
__**Klasa 1A**__

>>> `          UCZNIOWIE           `

**1**. Błaszczyk Oskar
**2**. Brom Kamil
**3**. Cichoń Marcel
**4**. Giza Klaudiusz
**5**. Guzek Łukasz
**6**. Hordejuk Jakub
**7**. Jęda Jakub
**8**. Josef Gracjan
**9**. Juranek Bartosz
**10**. Kapler Jakub
**11**. Klama Norbert
**12**. Koliński Igor
**13**. Koronkiwicz Kacper
**14**. Korpacki Mateusz
**15**. Kuraj Alan
**16**. Kurant Miłosz
**17**. Kuś Kacper
**18**. Lazarewicz Przemysław
**19**. Łoś Oliwier
**20**. Mikołajczyk Ksawier
**21**. Montak Mikołaj
**22**. Musiał Przemysław
**23**. Nawrocki Bartłomiej
**24**. Nowak Igor
**25**. Orłowski Arkadiusz
**26**. Przybysz Dawid
**27**. Rusin Wiktor
**28**. Rusiński Nikolas
**28**. Stefański Oskar
**29**. Wisielewski Oliwier
**30**. Wilczak Jastin
"""

#dziennik_image ="https://imgur.com/XXdjEGm"

finalper = False
tab = ["Kacper","Marcin","Łukasz"]

intents = discord.Intents().all()
client = commands.Bot(command_prefix='?', intents=intents)
intents.members = True

income_text = "𝙄𝙉𝘾𝙊𝙈𝙀 𝙍𝙀𝙈𝙄𝙉𝘿𝙀𝙍"

income_embed=discord.Embed(title=f'{income_text}\nDzień 2 wojny', description=">>> t\nt\nt", color=0xffff00)
income_embed.set_thumbnail(url="https://i.imgur.com/ZNsZsWs.png")
income_embed.set_image(url="https://i.imgur.com/KkxeKkh.png")
warstart = discord.Embed(title="Warstart Reminder 🗺️", description="GET ON NOW!", color=0x2b2d99)
warstart.set_thumbnail(url="https://i.imgur.com/z56YWQz.png")
warstart.set_image(url="https://i.imgur.com/BcSKgBX.png")

warstart_remind = discord.Embed(title="Warstart Reminder 🗺️", description="Warstart in 5 minutes, GET ON!", color=0x2b2d99)
warstart_remind.set_thumbnail(url="https://i.imgur.com/z56YWQz.png")
warstart_remind.set_image(url="https://i.imgur.com/BcSKgBX.png")
# [BOT EVENTS]
papajs = ["https://i.imgur.com/yGmCZM3.jpeg","https://i.imgur.com/hLTWsLY.png","https://i.imgur.com/l9bvw0Y.jpeg","https://i.imgur.com/uEgERWr.jpeg","https://i.imgur.com/uVpeEot.jpeg","https://i.imgur.com/VErQ7YZ.jpeg"]
status = cycle(["?pomoc", "?gayrate", "?ship", "Chemia Nowej Ery", "Fortnite", "wstępnej z twoją starą", "nie mogę oddychać"])

@client.event
async def on_guild_role_update(before, after):
    print(f'{before}, {after}')

@client.event
async def on_member_join(ctx):
    # 1600 800
    member_name = ctx.name
    card = Image.open("welcome_base.jpg")
    draw = ImageDraw.Draw(card)
    font = ImageFont.truetype("RussoOne-Regular.ttf", size=150)
    text = "Witamy Cię!"
    member_name = str(ctx.name)
    draw.text((1600,300), text, (0,0,0), anchor="ms", font=font)
    draw.text((1600,500), member_name, (255,0,0) , anchor="ms", font=font)
    asset = ctx.display_avatar
    data = BytesIO(await asset.read())
    pfp = Image.open(data)
    pfp = pfp.resize((553,553))
    card.paste(pfp,(210,200))
    card.save("welcome.jpg")
    dater = str(datetime.now())
    now = dater[:-10]
    with open("./bot_database/channel_config.json", "r") as f:
        channels = json.load(f)
    welcome = client.get_channel(int(channels["welcome"]))
    with open("./bot_database/members.json", "r") as r:
        members = json.load(r)
    if str(ctx.id) in members:
        again=discord.Embed(title=f"Witamy cie ponownie {ctx.mention} na maszym serwerze",description=f"Zajrzyj na #general!")
        await welcome.send(embed=again)
        await welcome.send(file= discord.File("welcome.jpg"))
    if str(ctx.id) not in members:
        indf = ctx.id
        members.update({indf:{}})
        members[indf].update({"joining_date":now})
        with open("./bot_database/members.json", "w") as t:
            json.dump(members, t, indent=1)
        joined= discord.Embed(title=f"Witamy cie {ctx.mention} na maszym serwerze",description=f"Zajrzyj na #general!")
        await welcome.send(embed=joined)
@client.event  
async def on_ready():
  synced = await client.tree.sync()
  for i in os.listdir("./cogs"):
      if i.endswith(".py"):
          await client.load_extension(f"cogs.{i[:-3]}")
  change_status.start()
  with open("./bot_database/channel_config.json", "r") as f:
        channels = json.load(f)
  general = client.get_channel(channels['general'])
  logs = client.get_channel(channels['logs'])
  await client.change_presence(activity=discord.Game(name="?pomoc"))
  print("Bot został włączony")
  embed=discord.Embed(title="FairBOT Logs",description="Operacja: Bot został poprawnie włączony",color=0x00ff00)
  now = datetime.now()
  test_time=str(now.weekday())
  week_day = str(now.weekday())
  test_now = str(datetime.now())
@tasks.loop(seconds=20)

async def change_status():
    
# [JSON DATA] ---------------------------------------------------------

  with open("./bot_database/channel_config.json", "r") as f:
    channels = json.load(f)
  with open("./bot_database/clan.json", "r") as f:
    clan_config = json.load(f)
  with open("./bot_database/income.json", "r") as f:
    income_hours = json.load(f)
  with open("./bot_database/weekdays.json", "r") as f:
    weekdays = json.load(f)
  with open("./bot_database/clan.json", "r") as t:
    clan_1 = json.load(t)
    hours = clan_1["income_hours"]
    hour1 = hours[0]
    hour2 = hours[1]
    hour3 = hours[2]
    hour4 = hours[3]
    hour5 = hours[4]
    hour6 = hours[5]
    
# [CHANNELS DEFINE] ---------------------------------------------------

  income_channel = client.get_channel(int(channels['income_channel']))
  general = client.get_channel(int(channels['general']))
  print("hej")

# [BOT STATUS] --------------------------------------------------------

  await client.change_presence(activity=discord.Game(next(status)))
    
# [DATETIME] ----------------------------------------------------------

  # [RAW DATETIME DATA]
    
  today_to_week = datetime.now() # 2012 13-06 13:00
  today_to_hour = str(datetime.now())
  raw_week = today_to_week.weekday()

  # [WEEK VARIABLES DEFINE]
    
  week = int(raw_week) # 1
  week_str = str(raw_week) # "1"
  weekday = weekdays[week_str] # poniedziałek etc.
  # [HOUR]
  now = str(today_to_hour[11:-10]) # 00:00

# [OTHERS] ------------------------------------------------------------

  warstart_day = int(clan_config['warstart'])
  warday = int(clan_config['war_day'])
  next_income = clan_config['next_income']

# [INCOME FREE DAYS] --------------------------------------------------

  dic_1 = {"05:00":True,"09:00":False,"13:00":False,"17:00":False,"21:00":False,"01:00":True}
  if week==0:
   if warstart_day == 2:
    with open("./bot_database/income.json", "w") as w:
        dic_2 = {"05:00":False,"09:00":False,"13:00":False,"17:00":False,"21:00":False,"01:00":False}
        income_hours["poniedziałek"].update(dic_1)
        income_hours["wtorek"].update(dic_2)
        json.dump(income_hours, w, indent=1)
   else:
    with open("./bot_database/income.json", "w") as w:
        income_hours["poniedziałek"].update(dic_1)
        json.dump(income_hours, w, indent=1)
    
# [EMBEDS] ------------------------------------------------------------

  income_text = "𝙄𝙉𝘾𝙊𝙈𝙀 𝙍𝙀𝙈𝙄𝙉𝘿𝙀𝙍"
  income_embed=discord.Embed(title=f'{income_text}\nDzień {warday} wojny', description="Zbierz income", color=0xffff00)
  income_embed.set_thumbnail(url="https://i.imgur.com/ZNsZsWs.png")
  income_embed.set_image(url="https://i.imgur.com/KkxeKkh.png")
  warstart = discord.Embed(title="Warstart Reminder 🗺️", description="GET ON NOW!", color=0x2b2d99)
  warstart.set_thumbnail(url="https://i.imgur.com/z56YWQz.png")
  warstart.set_image(url="https://i.imgur.com/BcSKgBX.png")
  warstart_remind = discord.Embed(title="Warstart Reminder 🗺️", description="Warstart in 5 minutes, GET ON!",   color=0x2b2d99)
  warstart_remind.set_thumbnail(url="https://i.imgur.com/z56YWQz.png")
  warstart_remind.set_image(url="https://i.imgur.com/BcSKgBX.png")

# [TIME REMINDERS] ---------------------------------------------------

  # [5 minutes before warstart] 
    
  if now=="08:55" and week==warstart_day:
    await income_channel.send("<@&1035829747298619452>")
    await income_channel.send(embed=warstart_remind)
    await asyncio.sleep(60)

  # [warstart or normal income 09:00]

  if income_hours[weekday]["09:00"] == True and now == hour2:
    if week==warstart_day:
     await income_channel.send("<@&1035829747298619452>")
     await income_channel.send(embed=warstart)
     with open("./bot_database/clan.json", "w") as f:
        clan_config.update({"next_income":"13:00"})
        json.dump(clan_config, f, indent=1)
     await asyncio.sleep(60)
    else:
     await income_channel.send("<@&1035829747298619452>")
     await income_channel.send(embed=income_embed)
     with open("./bot_database/clan.json","w") as f:
        clan_config.update({"next_income":"13:00"})
        json.dump(clan_config, f, indent=1)
     await asyncio.sleep(60)
        
  # [normal income 13:00]

  if income_hours[weekday]["13:00"] == True and now == hour3:
    await income_channel.send("<@&1035829747298619452>")
    await income_channel.send(embed=income_embed)
    with open("./bot_database/clan.json", "w") as f:
        clan_config.update({"next_income":"17:00"})
        json.dump(clan_config, f, indent=1)
    await asyncio.sleep(60)
    
  # [normal income 17:00]

  if income_hours[weekday]["17:00"] == True and now == hour4:
    await income_channel.send("<@&1035829747298619452>")
    await income_channel.send(embed=income_embed)
    with open("./bot_database/clan.json", "w") as f:
        clan_config.update({"next_income":"21:00"})
        json.dump(clan_config, f, indent=1)
    await asyncio.sleep(60)
    
  # [Codzienny posting papieża]

  if now=="21:37":
    await general.send(random.choice(papajs))
    await asyncio.sleep(60)
    
  # [normal income 21:00]

  if income_hours[weekday]["21:00"] == True and now == hour5:
    await income_channel.send("<@&1035829747298619452>")
    await income_channel.send(embed=income_embed)
    with open("./bot_database/clan.json", "w") as f:
        clan_config.update({"next_income":"01:00"})
        json.dump(clan_config, f, indent=1)
    await asyncio.sleep(60)
  # [next day]

  if now=="00:00" and week != 0:
    warday = warday + 1
    clan_config.update({"war_day":warday})
    with open("./bot_database/clan.json", "w") as f:
        json.dump(clan_config, f, indent=1)
    await asyncio.sleep(60)
    
  # [normal income 01:00]
    
  if income_hours[weekday]["01:00"] == True and now == hour6:
    await income_channel.send("<@&1035829747298619452>")
    await income_channel.send(embed=income_embed)
    with open("./bot_database/clan.json", "w") as f:
        clan_config.update({"next_income":"05:00"})
        json.dump(clan_config, f, indent=1)
    await asyncio.sleep(60)
    
  # [normal income 05:00]
  if income_hours[weekday]["05:00"] == True and now==hour1:
    await income_channel.send("<@&1035829747298619452>")
    await income_channel.send(embed=income_embed)
    with open("./bot_database/clan.json", "w") as f:
        clan_config.update({"next_income":"09:00"})
        json.dump(clan_config, f, indent=1)
    await asyncio.sleep(60)

    
#@client.event
#async def on_message(ctx):
    #if ctx.content.startswith('Gej'):
       #channel = ctx.channel
       #await channel.send("Sam jesteś gejem")

#@client.event
#async def on_command_error(ctx, error):
    #if isinstance(error, commands.MissingRequiredArgument):
        #await ctx.send('Zabrakło argumentu :rolling_eyes:.')
    #if isinstance(error,commands.MissingPermissions):
     #channel = client.get_channel(1025734128303345734)
        #await ctx.send(ctx.author.mention+" nie masz uprawnień 💀")
     #if isinstance(error, commands.CommandNotFound):
       #return
        #embed=discord.Embed(title="FairBOT Logs", description=f'Operacja: {ctx.author.mention} próbował wpisać komendę administratora')
        #embed.set_thumbnail(url="https://i.imgur.com/CDVo1v9.png")
        #await channel.send(embed=embed)

      #embed=discord.Embed(title="FairBOT Logs", description="Operacja: @zyzz próbował wpisać komendę administratora")
#embed.set_thumbnail(url="https://i.imgur.com/CDVo1v9.png")
#embed.add_field(name="Komenda:", value="?clear 5", inline=False)
#await ctx.send(embed=embed)
      
# [BOT COMMANDS]
@client.tree.command(name="hej", description='Testowa komenda bota "Hello World"')
async def hej(interaction: discord.Interaction):
    await interaction.response.send_message("Hello World")
    
@client.command()
@commands.has_permissions(administrator=True)
async def clan(ctx, key=None, ide=None):
   if key!=None and ide!=None:
    with open("./bot_database/income_config.json","r") as f:
        clan = json.load(f)
        clan.update({key:ide})
    with open("./bot_database/income_config.json", "w") as r:
        json.dump(clan, r, indent=1)
        await ctx.send("Zapisano zmianę")
   if key=="help":
    pass

@client.command()
@commands.has_permissions(administrator=True)
async def change_channel(ctx, channel=None, channel_id=None):
    if channel!=None and channel_id!=None:
     raw_id = int(channel_id)
     with open("./bot_database/channel_config.json","r") as f:
         channels=json.load(f)
         channels.update({channel:channel_id}) 
         with open("./bot_database/channel_config.json", "w") as r:
             json.dump(channels, r, indent=1)   
         await ctx.send("Pomyślnie zmieniono kanał")
    if channel=="help":
     await ctx.send("pomoc")

@client.command()
async def acc(ctx, member:discord.Member):
        member_id = str(member.id)
        avatar = member.display_avatar
        user =  client.get_user(member_id)
        #acc_age = str(user.created_at)
        with open("./bot_database/members.json", "r") as f:
            data = json.load(f)
        coins = data[member_id]["coins"]
        joined_date = data[member_id]["joined_date"]

        embed=discord.Embed(title=f'{member.name} Account!', description="_ _", color=0x452673)
        embed.set_author(name="member.name", icon_url=avatar)
        embed.set_thumbnail(url=avatar)
        embed.add_field(name="Joined to server:", value=joined_date, inline=True)
        embed.add_field(name="Account created at:", value="acc_age", inline=True)
        embed.add_field(name="Coins:", value=coins, inline=True)
        await ctx.send(embed=embed)

@client.command()
async def weekday(ctx):
    today = datetime.now()
    week_day = int(today.weekday())
    week = week_day + 1
    await ctx.send(f'Dzień tygodnia: {week}')
@client.command()
async def json_test(ctx):
    with open ("./bot_database/names.json", "w") as f:
        json.dump(papajs, f)
    with open ("./bot_database/names.json", "r") as f:
        final = json.load(f)
        await ctx.send(final)

@client.command()
async def dziennik(ctx):
    if ctx.channel.id == 1024374980663857242:
     channel = client.get_channel(1024374980663857242)
     await channel.send(diary)
    else:
     channel = client.get_channel(1024374980663857242)
     await channel.send(diary)
     await ctx.send("Ze względów bezpieczeństwa ta wiadomość pojawiła sie w sekcji klasowej")

@client.command()
@has_permissions(administrator=True)
async def adm_dziennik(ctx):
    await ctx.send(dziennik_image)
    await ctx.send(diary)     
@client.command()
async def database_name(ctx): 
    res = cur.execute("SELECT name FROM sqlite_master")
    name = str(res.fetchone())
    await ctx.send(name[2:-3])

@client.command()
async def add_user(ctx, id, user, coins, gems):
    data = (id, user, coins, gems)
    cur.executemany("INSERT INTO users VALUES(?, ?, ?, ?)", (data,))
    con.commit()
    await ctx.send("Dodano")

@client.command()
async def delete(ctx, index):
    cur.execute(f'DELETE FROM currency WHERE rowid=?', (index))
    await ctx.send("Usunięto")
@client.command()
async def check(ctx):
       table_select = "SELECT * FROM users"
       res = cur.execute(table_select)
       records = res.fetchall()
       for row in records:
        await ctx.send(f'Id: {row[0]}')
        await ctx.send(f'User: {row[1]}')
        await ctx.send(f'Coins: {row[2]}')
        await ctx.send(f'Gems: {row[3]}')
        await ctx.send("_ _")

@client.command()
async def check_user(ctx, user):
    table_select = "SELECT * FROM users where user= ?"
    cur.execute(table_select, (user,))
    records = cur.fetchall()
    for row in records:
      await ctx.send(f'User: {row[1]}')
      await ctx.send(f'Coins: {row[2]}')
      await ctx.send(f'Gems: {row[3]}')
@client.command()
async def base(ctx):
     res = cur.execute("SELECT * FROM users")
     base = res.fetchall()
     await ctx.send(base)

@client.command()
async def wstawaj(ctx):
  await ctx.send("Wstałem")

@client.command()
async def set_bot_status(ctx, *,status):
    await client.change_presence(activity=discord.Game(name=status))

@client.command()
async def losowa_liczba(ctx):
    await ctx.send(random.randint(0,99))

@client.command()
async def gayrate(ctx, text: discord.User):
  tag_user = str(text)
  user = tag_user[:-5]
  with open("./bot_database/gayrates.json") as f:
        gayrates = json.load(f)
  base_per= str(random.randint(0,100))+"%"
  if user in gayrates:
        per = gayrates[user]
        embed = discord.Embed(title="Sprawdzamy:",description=text)
        embed.add_field(name="Jest gejem w:", value=per, inline=False)
        await ctx.send(embed=embed)
  else:
        embed = discord.Embed(title="Sprawdzamy:",description=text)
        embed.add_field(name="Jest gejem w:",value=base_per, inline=False)
        await ctx.send(embed=embed)
        gayrates.update({user:base_per})
        with open("./bot_database/gayrates.json","w") as f:
            json.dump(gayrates,f)

@client.command()
async def ship(ctx, ship1, ship2):
 if ship1==ship2:
  await ctx.send("Nie możesz tego zrobić pajacu 🤨")
 else:
  per = random.randint(0,100)
  love = str(per) + "%"
  if per<40 or per==40 or per==0:
    desc = "Zimno 🧊"
  if per>40 and per<50:
    desc = "Koleżeńsko 😅"
  if per>50 and per<80:
    desc = "Iskrzy ✨"
  if per>80:
    desc = "Gorąco 🔥"
  embed=discord.Embed(title="Ship:", description= ship1 + " ❤️ " + ship2, color=0xff00ff)
  embed.add_field(name="Poziom miłości:", value=love + " **(" + desc + ")**", inline=False)
  await ctx.send(embed=embed)

@client.command()
async def pomoc(ctx):
  embed=discord.Embed(title="Komendy",  
  description="Komendy do bota Fair", 
  color=0x7b00ff)
  embed.add_field(name="?losowa_liczba", value="losuje liczbę", 
  inline=False)
  embed.add_field(name="?siema", value="witasz się z botem", inline=False)
  embed.add_field(name="?ship <osoba> <osoba>",value="sprawdza poziom miłości",inline=False)
  embed.add_field(name="?gayrate <osoba>", value="sprawdza poziom geja", inline=False)
  embed.add_field(name="?change_nick <osoba> <nowy_nick>",value="zmienia nick",inline=False)
  embed.add_field(name="?clear <liczba>", value="wyczyszcza ilość wiadomości", inline=False)
  embed.add_field(name="?walter", value="wysyła Włodzimierza", inline=False)
  embed.set_footer(text="okienko można wywołać za pomocą komendy ?pomoc")
  await ctx.send(embed=embed)
                       
@client.command()
@has_permissions(administrator=True)
async def ban(ctx, member : discord.Member, reason=None):
    channel = client.get_channel(1025734128303345734)
    if reason == None:
        await ctx.send(f"Ej {ctx.author.mention}, Musisz dać powód!")
    else:
        messageok = f"Dostałeś bana z {ctx.guild.name} z powodu: {reason}"
        await member.send(messageok)
        await member.ban(reason=reason)
        embed=discord.Embed(title="FairBOT Logs", description="Zbanowanie użytkownika",color=0xff0000)
        embed.set_thumbnail(url="https://static.wikia.nocookie.net/pixel-gun-3d/images/9/9b/Banhammerbig.png/revision/latest?cb=20200728091619")
        embed.add_field(name="Banujący:", value=ctx.author.mention)
        embed.add_field(name="Zbanowany:", value=member.mention)
        await channel.send(embed=embed)

@client.command()
@has_permissions(administrator=True)
async def kick(ctx, member : discord.Member, reason=None):
    channel = client.get_channel(1025734128303345734)
    if reason == None:
        await ctx.send(f"Ej {ctx.author.mention}, Musisz dać powód!")
    else:
        messageok = f"Zostałeś wyrzucony {ctx.guild.name} z powodu: {reason}"
        await member.send(messageok)
        await member.kick(reason=reason)
        embed=discord.Embed(title="FairBOT Logs", description="Operacja: Wyrzucenie użytkownika",color=0xff0000)
        embed.add_field(name="Wyrzucający:", value=ctx.author.mention)
        embed.add_field(name="Wyrzucony:", value=member.mention)
        await channel.send(embed=embed)

@client.command(pass_context = True)
@has_permissions(administrator=True)
async def mute(ctx, member: discord.Member):
        role = discord.utils.get(ctx.guild.roles, name="muted")
        await member.add_roles(role)
        await ctx.send("Dodano rangę")

@client.command(pass_context = True)
@has_permissions(administrator=True)
async def unmute(ctx, member: discord.Member):
        role = discord.utils.get(ctx.guild.roles, name="muted")
        await member.remove_roles(role)
        await ctx.send("Usunięto rangę")
        embed=discord.Embed(title="Permission Denied.", description="You don't have permission to use this command.", color=0xff00f6)
        await bot.say(embed=embed)

@client.command()
@has_permissions(administrator=True)
async def unban(ctx, *, member):
    banned_users = await ctx.guild.bans()
    member_name, member_discriminator = member.split('#')
    for ban_entry in banned_users:
        user = ban_entry.user

        if (user.name, user.discriminator) == (member_name,
                                               member_discriminator):
            await ctx.guild.unban(user)
            await ctx.send(f'Odbanowano {user.mention}')
            return

@client.command()                      
async def test(ctx):
    await ctx.send(ctx.author.mention)
    member_name, member_discriminator = member.split('#')
    for ban_entry in banned_users:
       user = ban_entry.user
       if (user.name, user.discriminator) == (member_name, member_discriminator):
            await ctx.guild.unban(user)
            await ctx.send(f'Unbanned {user.mention}')

@client.command()
async def walter(ctx):
    await ctx.send(random.choice(walters))
    await ctx.reply("To ja")

sex_embed = discord.Embed(title="Wybierz płeć", description="🙋 Chłopak\n🙋‍♀️ Dziewczyna\n🚁 Helikopter T-34 ")
@client.command()
@commands.has_permissions(administrator=True)
async def react1(ctx):
    with open("./bot_database/channel_config.json", "r") as f:
              channel_config = json.load(f)
    with open("./bot_database/self_role_config.json", "r") as r:
              self_config = json.load(r)
    channel = client.get_channel(channel_config["self_role"])
    message = await channel.send(embed=sex_embed)
    with open("./bot_database/self_role_config.json", "w") as w:
              self_config["gender"] = message.id
              json.dump(self_config, w, indent=1)
    await message.add_reaction("🙋")
    await message.add_reaction("🙋‍♀️")
    await message.add_reaction("🚁")
ping_embed = discord.Embed(title="Powiadomienia", description="1️⃣ Warstart\n2️⃣ Income\n3️⃣ Clan Raid")

@client.command()
@commands.has_permissions(administrator=True)
async def react2(ctx):
    with open("./bot_database/channel_config.json", "r") as f:
        channel_config = json.load(f)
    with open("./bot_database/self_role_config.json", "r") as r:
        self_config = json.load(r)
    channel = client.get_channel(channel_config["self_role"])
    message = await channel.send(embed=ping_embed)
    with open("./bot_database/self_role_config.json", "w") as f:
        self_config["ping"] = message.id
        json.dump(self_config, f, indent=1)
    await message.add_reaction("1️⃣")
    await message.add_reaction("2️⃣")
    await message.add_reaction("3️⃣")
    
reg = discord.Embed(title="Zatwierdź regulamin", description="Zareaguj poniżej aby się zweryfikować")
@client.command()
@commands.has_permissions(administrator=True)
async def statute(ctx):
    with open("./bot_database/channel_config.json", "r") as f:
        channel_config = json.load(f)
    with open("./bot_database/self_role_config.json", "r") as r:
        self_config = json.load(r)
    channel = client.get_channel(channel_config["statute"])
    message = await channel.send(embed=reg)
    with open("./bot_database/self_role_config.json", "w") as w:
        self_config["statute"] = message.id
        json.dump(self_config, w, indent=1)
    await message.add_reaction("✔️")
@client.command()
async def walter_gallery(ctx):
    for i in walters:
      time.sleep(0.5)
      await ctx.send(i)
        
@client.command()
async def avatar(ctx, member: discord.Member=None):
    if member==None:
        member=ctx.author
    pfp = member.display_avatar
    embed=discord.Embed(title=f'{member.name} Avatar', description="t",color=discord.Colour.blue())
    embed.set_image(url=pfp)
    await ctx.send(embed=embed)
    
@client.command(pass_context=True)
async def change_nick(ctx, member: discord.Member, nick):
    await member.edit(nick=nick)
    await ctx.send(f'Zmieniono nick!')
    
@client.command()
async def income_title(ctx, *, data):
    with open("./bot_database/income_config.json", "r") as f:
        income_config = json.load(f)
        income_config["description"] = data
    with open("./bot_database/income_config.json", "w") as f:
        json.dump(income_config, f, indent=1)
@client.event
async def on_raw_reaction_add(payload):
    with open("./bot_database/self_role_config.json", "r") as r:
        self_role = json.load(r)
        gender = self_role["gender"]
        ping = self_role["ping"]
        statute = self_role["statute"]
    message_id = payload.message_id
    guild = discord.utils.find(lambda g: g.id==payload.guild_id, client.guilds)
    member = discord.utils.find(lambda g: g.id==payload.user_id, guild.members)
    boy = discord.utils.get(guild.roles, name="Chłopak")
    girl = discord.utils.get(guild.roles, name="Dziewczyna")
    war = discord.utils.get(guild.roles, name="Warstart")
    income = discord.utils.get(guild.roles, name="Income")
    clan_raid = discord.utils.get(guild.roles, name="Clan Raid")
    helikopter = discord.utils.get(guild.roles, name="Helikopter T-34")
    member_role = discord.utils.get(guild.roles, name="Member")
    if payload.emoji.name == "🙋" and message_id == gender:
            await member.add_roles(boy)
    if payload.emoji.name == "🙋‍♀️" and message_id == gender:
            await member.add_roles(girl)
    if payload.emoji.name == "🚁" and message_id == gender:
            await member.add_roles(helikopter)
    if payload.emoji.name == "1️⃣" and message_id == ping:
            await member.add_roles(war)
    if payload.emoji.name == "2️⃣" and message_id == ping:
            await member.add_roles(income)
    if payload.emoji.name == "3️⃣" and message_id == ping:
            await member.add_roles(clan_raid)
    if payload.emoji.name == "✔️" and message_id == statute:
            await member.add_roles(member_role)

@client.event
async def on_raw_reaction_remove(payload):
    with open("./bot_database/self_role_config.json", "r") as r:
        self_role = json.load(r)
        gender = self_role["gender"]
        ping = self_role["ping"]
        statute = self_role["statute"]
    message_id = payload.message_id
    guild = discord.utils.find(lambda g: g.id==payload.guild_id, client.guilds)
    member = discord.utils.find(lambda g: g.id==payload.user_id, guild.members)
    boy = discord.utils.get(guild.roles, name="Chłopak")
    girl = discord.utils.get(guild.roles, name="Dziewczyna") 
    helikopter= discord.utils.get(guild.roles, name="Helikopter T-34")
    war = discord.utils.get(guild.roles, name ="Warstart")
    income = discord.utils.get(guild.roles, name="Income")
    clan_raid = discord.utils.get(guild.roles, name= "Clan Raid")
    member_role = discord.utils.get(guild.roles, name="Member")
    if payload.emoji.name == "🙋" and message_id == gender:
            await member.remove_roles(boy)
    if payload.emoji.name == "🙋‍♀️" and message_id == gender:
            await member.remove_roles(girl)
    if payload.emoji.name == "🚁" and message_id == gender:
            await member.remove_roles(helikopter)
    if payload.emoji.name == "1️⃣" and message_id == ping:
            await member.remove_roles(war)
    if payload.emoji.name == "2️⃣" and message_id == ping:
            await member.remove_roles(income)
    if payload.emoji.name == "3️⃣" and message_id == ping:
            await member.remove_roles(clan_raid)
@client.command()
async def photo_text(ctx, font_size: int = 50, *, text = "Brak Tekstu"):
  photo = Image.open("white.jpg")
  photo = photo.resize((1000,1000))
  draw = ImageDraw.Draw(photo)
  font = ImageFont.truetype("RussoOne-Regular.ttf", font_size)
  draw.text((500,500), text, (0,0,0), anchor="ms", font=font)
  photo.save("new_photo.jpg")
  await ctx.send(file = discord.File("new_photo.jpg"))
@client.command()
@commands.has_permissions(administrator=True)
async def embed(ctx, number: int):
  if number==1:
    await ctx.send("<@&1035829747298619452>")
    await ctx.send(embed=income_embed)
  if number==2:
    await ctx.send("<@&1035829747298619452>")
    await ctx.send(embed=warstart)
  if number==3:
    await ctx.send("<@&1035829747298619452>")
    await ctx.send(embed=warstart_remind)
  if number==69:
    await ctx.send("1 - 4hours income, 2 - warstart 9, 3 - warstart 5 minutes before reminder")
@client.command()
async def snipe(ctx, ds_member: discord.Member):
   try:
    username = str(ds_member)
    with open("./bot_database/last_messages.json", "r") as f:
        last_messages = json.load(f)
    user_last_message = last_messages[username[:-5]]
    await ctx.send(user_last_message)
   except:
    await ctx.send("Ten użytkownik nie wysłał żadnej wiadomości!")
                                   
@client.event
async def on_message(ctx):
    if ctx.channel.id in [1049410779738284083,1050502781032738909] and ctx.author.name != "FairBOT":
        openai.api_key = "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"
        
        message = await ctx.channel.send("Chwileczkę przetwarzam twoje polecenie...")
        response = openai.Completion.create(model="text-davinci-003",prompt=ctx.content,temperature=0.7,max_tokens=3521,top_p=1,frequency_penalty=0,presence_penalty=0)
        info = dict(response)
        print(type(info))
        await message.edit(content=(info["choices"][0]["text"]))
    if ctx.content=="@everyone":
        words = ["Co mnie oznaczasz ty kurwo jebana","Wyjebać ci kopa za tego pinga","Rozjebie ci dupe zaraz jak sie nie uspokoisz", "O ty kurwo", "Morda pod fiutem", "Zaraz cię zgwałce"]
        await ctx.reply(random.choice(words))
    if ctx.author.name not in ["Grzanka", "FairBOT", "FairSecurity"]:
     username = ctx.author.name
     with open("./bot_database/messages.json", "r") as f:
        user_dic = json.load(f)
        if username in user_dic:
          messages_number = user_dic[username]
          messages_number = messages_number + 1
          with open("./bot_database/messages.json", "w") as f:
            user_dic.update({username:messages_number})
            json.dump(user_dic, f, indent=1)
        else:
          with open("./bot_database/messages.json", "w") as f:
            user_dic.update({username:1})
            json.dump(user_dic, f, indent=1)
     await client.process_commands(ctx)
    if ctx.content=="bajo jajo":
        await ctx.channel.send("Bajo jajo ty chuju jebany na inowrocławskiej jesteś, ja ci zaraz dam bajo jajo kurwa, zaraz cię ściągnę i ci chuju do dupy dokopię kurwa bajo jajo pierdolone")
        
        
@client.tree.command(description="Income List")
async def income_list(ctx: discord.Interaction):
    embed = discord.Embed(title="Income List", description="❗ Jeżeli dana godzina jest **True**, wtedy o tej godzinie będzie sie pojawiać powiadomienie\n❓Jeżeli dana godzina jest **False**, wtedy o tej godzinie nie będzie pojawiać się powiadomienie")
    with open("./bot_database/income.json", "r") as r:
        income = json.load(r)
    with open("./bot_database/clan.json", "r") as r:
        clan = json.load(r)
    pon1 = income["poniedziałek"]["05:00"]
    pon2 = income["poniedziałek"]["09:00"]
    pon3 = income["poniedziałek"]["13:00"]
    pon4 = income["poniedziałek"]["17:00"]
    pon5 = income["poniedziałek"]["21:00"]
    pon6 = income["poniedziałek"]["01:00"]
    wt1 = income["wtorek"]["05:00"]
    wt2 = income["wtorek"]["09:00"]
    wt3 = income["wtorek"]["13:00"]
    wt4 = income["wtorek"]["17:00"]
    wt5 = income["wtorek"]["21:00"]
    wt6 = income["wtorek"]["01:00"]
    sr1 = income["środa"]["05:00"]
    sr2 = income["środa"]["09:00"]
    sr3 = income["środa"]["13:00"]
    sr4 = income["środa"]["17:00"]
    sr5 = income["środa"]["21:00"]
    sr6 = income["środa"]["01:00"]
    czw1 = income["czwartek"]["05:00"]
    czw2 = income["czwartek"]["09:00"]
    czw3 = income["czwartek"]["13:00"]
    czw4 = income["czwartek"]["17:00"]
    czw5 = income["czwartek"]["21:00"]
    czw6 = income["czwartek"]["01:00"]
    pt1 = income["piątek"]["05:00"]
    pt2 = income["piątek"]["09:00"]
    pt3 = income["piątek"]["13:00"]
    pt4 = income["piątek"]["17:00"]
    pt5 = income["piątek"]["21:00"]
    pt6 = income["piątek"]["01:00"]
    sob1 = income["sobota"]["05:00"]
    sob2 = income["sobota"]["09:00"]
    sob3 = income["sobota"]["13:00"]
    sob4 = income["sobota"]["17:00"]
    sob5 = income["sobota"]["21:00"]
    sob6 = income["sobota"]["01:00"]
    ndz1 = income["niedziela"]["05:00"]
    ndz2 = income["niedziela"]["09:00"]
    ndz3 = income["niedziela"]["13:00"]
    ndz4 = income["niedziela"]["17:00"]
    ndz5 = income["niedziela"]["21:00"]
    ndz6 = income["niedziela"]["01:00"]
    addn_wt = ""
    addn_sr = ""
    if clan["warstart"] == "1":
        addn_wt = "(Warstart)"
    if clan["warstart"] == "2":
        addn_sr = "(Warstart)"
    embed.add_field(name=f"📅 Poniedziałek", value=f'>>> 🕐 **05:00**: {pon1}\n🕐 **09:00**: {pon2}\n🕐 **13:00**: {pon3}\n🕐 **17:00**: {pon4}\n🕐 **21:00**: {pon5}\n🕐 **01:00**: {pon6}')
    embed.add_field(name=f"📅 Wtorek {addn_wt}", value=f'>>> 🕐 **05:00**: {wt1}\n🕐 **09:00**: {wt2}\n🕐 **13:00**: {wt3}\n🕐 **17:00**: {wt4}\n🕐 **21:00**: {wt5}\n🕐 **01:00**: {wt6}')
    embed.add_field(name=f"📅 Środa {addn_sr}", value=f'>>> 🕐 **05:00**: {sr1}\n🕐 **09:00**: {sr2}\n🕐 **13:00**: {sr3}\n🕐 **17:00**: {sr4}\n🕐 **21:00**: {sr5}\n🕐 **01:00**: {sr6}')
    embed.add_field(name=f"📅 Czwartek", value=f'>>> 🕐 **05:00**: {czw1}\n🕐 **09:00**: {czw2}\n🕐 **13:00**: {czw3}\n🕐 **17:00**: {czw4}\n🕐 **21:00**: {czw5}\n🕐 **01:00**: {czw6}')
    embed.add_field(name=f"📅 Piątek", value=f'>>> 🕐 **05:00**: {pt1}\n🕐 **09:00**: {pt2}\n🕐 **13:00**: {pt3}\n🕐 **17:00**: {pt4}\n🕐 **21:00**: {pt5}\n🕐 **01:00**: {pt6}')
    embed.add_field(name=f"📅 Sobota", value=f'>>> 🕐 **05:00**: {sob1}\n🕐 **09:00**: {sob2}\n🕐 **13:00**: {sob3}\n🕐 **17:00**: {sob4}\n🕐 **21:00**: {sob5}\n🕐 **01:00**: {sob6}')
    embed.add_field(name=f"📅 Niedziela", value=f'>>> 🕐 **05:00**: {ndz1}\n🕐 **09:00**: {ndz2}\n🕐 **13:00**: {ndz3}\n🕐 **17:00**: {ndz4}\n🕐 **21:00**: {ndz5}\n🕐 **01:00**: {ndz6}')
    embed.set_footer(text="Aby zmienić parametry użyj wbudowanych komend: \n/income\n/warstart")
    await ctx.response.send_message(embed=embed)

@client.tree.command(description="income config test")
@app_commands.choices(weekday=[
    Choice(name="poniedziałek", value="poniedziałek"),
    Choice(name="wtorek", value="wtorek"),
    Choice(name="środa", value="środa"),
    Choice(name="czwartek", value="czwartek"),
    Choice(name="piątek", value="piątek"),
    Choice(name="sobota", value="sobota"),
    Choice(name="niedziela", value="niedziela"),
],hour=[
    Choice(name= hour1, value="05:00"),
    Choice(name= hour2, value="09:00"),
    Choice(name= hour3, value="13:00"),
    Choice(name= hour4, value="17:00"),
    Choice(name= hour5, value="21:00"),
    Choice(name= hour6, value="01:00"),
    Choice(name="wszystkie godziny", value="all"),
],statement=[
    Choice(name="True", value="True"),
    Choice(name="False", value="False"),
])
async def income(ctx: discord.Interaction, weekday:str, hour:str, statement: str):
   member_id = ctx.user.id
   if member_id in [424502321800675328]:
    array = []
    with open("./bot_database/income.json", "r") as read_income:
        income = json.load(read_income)
    if hour == "all":
        if statement=="True":
            state=True
        if statement=="False":
            state=False
        all_dic = {"05:00":state,"09:00":state,"13:00":state,"17:00":state,"21:00":state,"01:00":state}
        with open("./bot_database/income.json", "w") as write_income:
            income[weekday].update(all_dic)
            json.dump(income, write_income, indent=1)
            await ctx.response.send_message("Pomyślnie zapisano zmiany")
    else:
        if statement=="True":
            state=True
        if statement=="False":
            state=False
        with open("./bot_database/income.json", "w") as write_income:
            income[weekday][hour] = state
            json.dump(income, write_income, indent=1)
            await ctx.response.send_message("Pomyślnie zapisano zmiany")
   else:
     await ctx.response.send_message("Nie masz uprawnień")

@client.tree.command(description="informacje o wojnie klanów")
async def clan_war(ctx: discord.Interaction):
    with open("./bot_database/clan.json", "r") as f:
        clan_config = json.load(f)
        time_from_json = clan_config["next_income"]
        warday = clan_config["war_day"]
        warstart = clan_config["warstart"]
        time = str(time_from_json)
    today_to_week = datetime.now()
    week = int(today_to_week.weekday())
    today_to_hour = str(datetime.now())
    now = str(today_to_hour[11:-10])
    FMT = "%H:%M"
    strp_time = str(datetime.strptime(time, FMT) - datetime.strptime(now, FMT))
    cuted_time = strp_time[-7:-3]
    hours = cuted_time[0:1]
    minutes = cuted_time[2:]
    minute_last = minutes[1]
    h_word = "godziny"
    m_word = "minut"
    hour_pass = True
    minute_pass = True
    embed = discord.Embed(title="Informacje o wojnie klanów", description=f'{str(warday)} dzień wojny')
    if hours=="1":
        h_word = "godzina"
    if hours=="0":
        hour_pass = False
    if minutes=="00":
        minute_pass = False
    if minutes=="01":
        minutes = "1"
        m_word = "minuta"
    if minute_last in ["2","3","4"]:
        if minutes.startswith("0"):
            minutes = minute_last
        m_word = "minuty"
    if minute_last in ["5","6","7","8","9"]:
        if minutes.startswith("0"):
            m_word = "minut"
            minutes = minute_last
        else:
            m_word = "minut"
    if week >= warstart:
     if minute_pass==False:
        embed.add_field(name="Czas do kolejnego income:", value=f'{hours} {h_word}')
     if hour_pass==False:
        embed.add_field(name="Czas do kolejnego income:", value=f'{minutes} {m_word}')
     if hour_pass==True and minute_pass==True:
        embed.add_field(name="Czas do kolejnego income:", value=f'{hours} {h_word} i {minutes} {m_word}')
    if week < warstart and warstart==1:
        embed.add_field(name="Wojna zacznie sie we wtorek", value=None)
    if week < warstart and warstart==2:
        embed.add_field(name="Wojna zacznie sie w środę", value=None)
     
    
    await ctx.response.send_message(embed=embed)
@client.tree.command(description="warstart change")
@app_commands.checks.bot_has_permissions(administrator=True)
@app_commands.choices(weekday=[
    Choice(name="wtorek", value=1),
    Choice(name="środa", value=2)
])
async def warstart(ctx: discord.Interaction, weekday: int):
   member_id = ctx.user.id
   if member_id in [424502321800675328]:
    with open("./bot_database/clan.json", "r") as f:
        clan_config = json.load(f)
        warstart = clan_config["warstart"]
    with open("./bot_database/clan.json", "w") as f:
        clan_config.update({"warstart":weekday})
        json.dump(clan_config, f, indent=1)
    await ctx.response.send_message("Pomyślnie zmienione dzień rozpoczęcia wojny")
   else:
    await ctx.response.send_message("Nie masz uprawnień")

@client.tree.command(description="Wykonaj task")
async def task(ctx: discord.Interaction):
    await ctx.response.send_message("Ta interakcja jest chwilowo niedostępna lub wyłączona przez administratora bota, spróbuj ponownie później")
    
@client.tree.command(description="Podium ludzi z największą ilością wiadomości")
async def top(ctx: discord.Interaction):
    with open("./bot_database/messages.json", "r") as f:
        user_dic = json.load(f)
    table = []
    users = []
    first_place = "🥇 **Brak danych**"
    second_place ="🥈 **Brak danych**"
    third_place = "🥉 **Brak danych**"
    for i in user_dic:
        table.append(user_dic[i])
        users.append(i)
    best_score_index = table.index(max(table))
    table_len = len(table)
    
    if table_len >= 1:
     best_user = users[best_score_index]
     best_score = max(table)
     first_place =f'🥇 **{best_user}**: {best_score}'
    if table_len >= 2:
     table.pop(best_score_index)
     users.pop(best_score_index)
     best_score_index = table.index(max(table))
     best_score = max(table)
     best_user = users[best_score_index]
     second_place = f'🥈 **{best_user}**: {best_score}'
    if table_len >= 3:
     table.pop(best_score_index)
     users.pop(best_score_index)
     best_score_index = table.index(max(table))
     best_score = max(table)
     best_user = users[best_score_index]
     third_place = f'🥉 **{best_user}**: {best_score}'
    
    podium = f'>>> {first_place}\n{second_place}\n{third_place}\n'
    
    embed=discord.Embed(title="Ranking", description="Ilość wysłanych wiadomości na tym serwerze")
    embed.add_field(name="Podium", value=podium)
    
    await ctx.response.send_message(embed=embed)

try:
  client.run("token")
except:
  print("[ Krytyczny Błąd ] - Token był offline dłużej niż 10 sekund")
#asyncio.run(main())
