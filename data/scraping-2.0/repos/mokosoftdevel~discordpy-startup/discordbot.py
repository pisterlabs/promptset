from discord.ext import commands
import os
import traceback
import random
import discord
import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials 
from datetime import datetime,timedelta,timezone
from discord.ext import tasks
import numpy as np
import cv2
import time
import openai

from dotenv import load_dotenv
load_dotenv()

gpt_secret_key = os.environ['OPEN_AI_SECRET']
gpt_orgnize = os.environ['OPEN_AI_ORGNIZE']

bot = commands.Bot(command_prefix='うんこ')
token = os.environ['DISCORD_BOT_TOKEN']


# うんこの受け答えlist
unko_messages = []
unko_dict = {} # ユーザー別やりとりリスト

# ねむいlist
unko_nemui = []

# おはよう
unko_ohayo = []

# massa
unko_massa = []

# 食べよ
unko_tabeyo = []

# schedule
unko_schedule = []

# slot list
unko_slot = []
unko_slot2 = []
custom_slot = []
kakuritu = 0.0045

# omikuji
unko_omikuji = []

# 削除リスト
unko_log = []
is_log_check = False

# 永続プロンプト
unko_prompt = []



# google spread sheet api 
sheet = os.environ['SHEETKEY']
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']    
credential = {
                "type": "service_account",
                "project_id": os.environ['SHEET_PROJECT_ID'],
                "private_key_id": os.environ['SHEET_PRIVATE_KEY_ID'],
                "private_key": os.environ['SHEET_PRIVATE_KEY'].replace('\\n', '\n'),
                "client_email": os.environ['SHEET_CLIENT_EMAIL'],
                "client_id": os.environ['SHEET_CLIENT_ID'],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url":  os.environ['SHEET_CLIENT_X509_CERT_URL']
             }
# print(credential)
credentials = ServiceAccountCredentials.from_json_keyfile_dict(credential, scope)
gc = gspread.authorize(credentials)
wb = gc.open_by_key(sheet)
sheet_messages = wb.worksheet('messages')
sheet_uranai = wb.worksheet('uranai')
sheet_ohayou = wb.worksheet('ohayou')
sheet_massa = wb.worksheet('massa')
sheet_tabeyo = wb.worksheet('tabeyo')
sheet_schedule = wb.worksheet('schedule')
sheet_slot = wb.worksheet('slot')
sheet_omikuji = wb.worksheet('omikuji')
sheet_log = wb.worksheet('log')

bot_channel_id = 738973128645935104
JST = timezone(timedelta(hours=+9), 'JS')


@tasks.loop(seconds=60)
async def loop():
    global unko_schedule
    now = datetime.now(JST).strftime('%H:%M')
    # print(now)
    for line in unko_schedule:
        if now == line[0]:
            channel = bot.get_channel(int(line[1]))
            print(line)
            await channel.send(line[2])
            if line[2] == 'うんこすろっと':
                await com_deka_slot(channel)
    if now == "00:00": 
        await func_all_reload()
    #if now == "13:45":
    #    channel = bot.get_channel(738973128645935104)
    #    ato = datetime(2022,11,27) - datetime.now()
    #    await channel.send("うんこbotが死ぬまであと "+str(ato)+"日...")

loop.start()

@tasks.loop(seconds=1)
async def loop_second():
    global unko_log
    global is_log_check
    now = datetime.utcnow()
    #print(now)
    #print(bot)
    if is_log_check:
        is_log_check = False
        for line in unko_log:
            #print(line[1])
            channel = bot.get_channel(int(line[1]))
            if channel is None:
                print("Channel is none")
                is_log_check = True
                return
            #print(now)
            #print(channel)
            #print(int(line[0]))
            msgs = []
            async for message in channel.history(limit=100):
                #print(message.content)
                if now < message.created_at:
                    continue
                td = now - message.created_at
                #print(message.content)
                #print(message.created_at)
                #print(td.seconds)
                if td.seconds >= int(line[0]) and td.days < 14:
                    msgs.append(message)
                    #print(message.content)
            try:
                await channel.delete_messages(msgs)
            except (ClientException, Forbidden, NotFound, HTTPException) as e:
                print(e)
                is_log_check = True

                    
                    
        is_log_check = True

loop_second.start()


@bot.event
async def on_command_error(ctx, error):
    orig_error = getattr(error, "original", error)
    error_msg = ''.join(traceback.TracebackException.from_exception(orig_error).format())
    #await ctx.send(error_msg)

@bot.event
async def on_message(message):
    await bot.process_commands(message)
    if message.author.bot:
        return
    if message.content != 'うんこ' and message.content.startswith('うんこ'):
        return
    global unko_messages
    for line in unko_messages:
        if line[0] == 'end':
            if message.content.endswith(line[1]):
                if int(line[3]) == 1:
                    await message.add_reaction(line[4])
                if len(line[2]) > 0:
                    await message.channel.send(line[2])
                return
        elif line[0] == 'find':
            if line[1] in message.content:
                if int(line[3]) == 1:
                    await message.add_reaction(line[4])
                if len(line[2]) > 0:
                    await message.channel.send(line[2])
                return
        


@bot.command()
async def ヘルプ(ctx):
    await ctx.send("""うんこって誰 : わいが返事するで

うんこねむい : 眠気をはかるで

うんこどう？ : うんこの状態を教えるで

うんこmassa : Massaを罵倒するで

うんこ何食べよ : 食べるものを提案するよ

うんこおはよう : 占い
""")

@bot.command(aliases=['だれ','だれ？','誰','誰？'])
async def com_dare(ctx):
    await ctx.send('わいや')


@bot.command(aliases=['ねむい','眠い'])
async def com_nemui(ctx):
    rand_int = random.randint(0,100)
    nemu_mes = ''
    global unko_nemui
    for line in unko_nemui:
        if int(line[0]) <= rand_int and rand_int <= int(line[1]):
            nemu_mes = line[2]
            break
    await ctx.send(f"{ctx.author.mention}"+' '+nemu_mes+' ('+str(rand_int)+')' )


@bot.command(aliases=['どう？','どう'])
async def ping(ctx):
    rand_int = random.randint(0,100)
    await ctx.send(f"{ctx.author.mention}"+' うんこのかんじは '+str(rand_int)+' やな' )


@bot.command(aliases=['massa','Massa','まっさ'])
async def com_massa(ctx):
    global unko_massa
    mes = random.choice(unko_massa)
    await ctx.send(mes)


@bot.command(aliases=['何食べよ','何食べよ？'])
async def com_tabeyo(ctx):
    global unko_tabeyo
    mes = random.choice(unko_tabeyo)
    await ctx.send(f"{ctx.author.mention}"+' '+mes)

@bot.command(aliases=['PromptAdd', 'PromptADD'])
async def com_promptadd(ctx, *args):
    global unko_prompt

    for item in args:
        unko_prompt.append(item)
    await ctx.send("promptを追加しました")

@bot.command(aliases=['PromptClean', 'PromptClear'])
async def com_promptclear(ctx):
    global unko_prompt
    unko_prompt.clear()
    await ctx.send("promptを初期化しました")

@bot.command(aliases=['PromptCheck'])
async def com_promptcheck(ctx):
    global unko_prompt

    prompt = ""
    for item in unko_prompt:
        prompt += item + "\n"
    await ctx.send("現在のpromptは以下の通りです\n"+prompt)



@bot.command(aliases=['AI','ＡＩ'])
async def com_ai(ctx, *args):
    global gpt_secret_key
    global gpt_orgnize
    global unko_dict

    if len(args) <= 0:
        await ctx.send('promptを指定してください')
        return
    
    user_name = ctx.author.display_name
    print(user_name)

    prompt = ""
    for item in args:
        prompt += item + "\n"
    prompt = prompt.rstrip()

    messanger_name = "You"
    prolist = ""
    if user_name in unko_dict:
        # リストを取得し追加する
        print("1")
        prolist = unko_dict[user_name]

        prolist_split = prolist.split("\n")
        if len(prolist_split) > 10:
            prolist_split = prolist_split[2:]
            prolist = "\n".join(prolist_split)

        prolist += "\n" + messanger_name + ": " + prompt + "\n" + "AI: "
    else:
        print("2")
        prolist = messanger_name + ": " + prompt + "\n" + "AI: "

    #print(prolist)

    com_prompt = ""
    for item in unko_prompt:
        com_prompt += item + "\n"

    
    #print(prompt)
    #prompt = com_prompt + "\n" + prolist
    print("prompt: "+ prompt)

    print("org:"+gpt_orgnize)
    print("key:"+gpt_secret_key)

    openai.organization = gpt_orgnize
    openai.api_key = gpt_secret_key

    #completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world!"}])
    #print("return")
    #print(completion)

    #print(openai.api_key)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        # prompt=prompt,
        messages = [
            {"role": "user", "content": prompt}
        ]
        #temperature=1
        #temperature=0.3,
        #max_tokens=2048,
        #top_p=1.0,
        #frequency_penalty=0.0,
        #presence_penalty=0.0
    )
    print(response)
    #texts = ''.join([choice['text'] for choice in response.choices])
    texts = response.choices[0].message.content
    print(texts)

    prolist += texts
    unko_dict[user_name] = prolist

    await ctx.send(texts)

@bot.command(aliases=['UserClear'])
async def com_ai_user_clear(ctx):
    global unko_dict
    user_name = ctx.author.display_name


@bot.command(aliases=['画像'])
async def com_image(ctx):
    img_red = np.zeros((1000, 1000, 3), np.uint8)
    img_red[:, :, 2] = 255
    #cv2.imwrite('tmp.png', img_red)
    #img = cv2.imread('tmp.png')

    msgs = ['omaera','zettaini','yurusanai','massa','tasukete','korosu','yurusu','unko','tabero','FAX','666','saintsaiya']

    for num in range(0, random.randint(3,12)):
        cv2.putText(img_red,
            text=random.choice(msgs),
            org=(random.randint(-100,1000), random.randint(-100,1000)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=random.uniform(0.5,8.0),
            color=(random.randint(200,255), 255, 255),
            thickness=random.randint(1,8),
            lineType=cv2.LINE_4)
    
    
    cv2.imwrite('tmp.png', img_red)
    await ctx.send(file=discord.File('tmp.png'))



@bot.command(aliases=['すろーっと','スローット'])
async def com_sloooot(ctx):
    global unko_slot

    mes1 = random.choice(unko_slot)
    mes2 = random.choice(unko_slot)
    mes3 = random.choice(unko_slot)
    mes4 = random.choice(unko_slot)
    mes5 = random.choice(unko_slot)
    mes6 = random.choice(unko_slot)
    mes7 = random.choice(unko_slot)
    mes8 = random.choice(unko_slot)
    mes9 = random.choice(unko_slot)
    mes10 = random.choice(unko_slot)
    await ctx.send(mes1+mes2+mes3+mes4+mes5+mes6+mes7+mes8+mes9+mes10)


@bot.command(aliases=['スロット１連','すろっと１連'])
async def com_slot(ctx):
    global unko_slot

    dice = list()
    for x in range(1,8):
        for y in range(1,8):
            for z in range(1,8):
                dice.append(100*x+10*y+z)
    print(dice)

    prob = list()
    for i in range(343):
        if dice[i] == 111:
            prob.append(0.0045)
        elif dice[i] == 222:
            prob.append(0.0045)
        elif dice[i] == 333:
            prob.append(0.0045)
        elif dice[i] == 444:
            prob.append(0.0045)
        elif dice[i] == 555:
            prob.append(0.0045)
        elif dice[i] == 666:
            prob.append(0.0045)
        elif dice[i] == 777:
            prob.append(0.0045)
        else:
            prob.append(0.9685/336)

    samples = np.random.choice(a=dice,size=1,p=prob)
    print(samples)

    for item in samples:
        n = item
        num = map(int, str(n))
        nums = list(num)
        print(nums)
        mes = unko_slot[nums[0]]+unko_slot[nums[1]]+unko_slot[nums[2]]
        await ctx.send(mes)

    #mes1 = random.choice(unko_slot)
    #mes2 = random.choice(unko_slot)
    #mes3 = random.choice(unko_slot)
    #await ctx.send(mes1+mes2+mes3)

@bot.command(aliases=['スロット３連','すろっと３連'])
async def com_slot7(ctx):
    global unko_slot

    dice = list()
    for x in range(1,8):
        for y in range(1,8):
            for z in range(1,8):
                dice.append(100*x+10*y+z)
    print(dice)

    prob = list()
    for i in range(343):
        if dice[i] == 111:
            prob.append(0.00045)
        elif dice[i] == 222:
            prob.append(0.00045)
        elif dice[i] == 333:
            prob.append(0.00045)
        elif dice[i] == 444:
            prob.append(0.00045)
        elif dice[i] == 555:
            prob.append(0.00045)
        elif dice[i] == 666:
            prob.append(0.00045)
        elif dice[i] == 777:
            prob.append(0.00045)
        else:
            prob.append(0.99685/336)

    samples = np.random.choice(a=dice,size=3,p=prob)
    print(samples)

    for item in samples:
        n = item
        num = map(int, str(n))
        nums = list(num)
        print(nums)
        mes = unko_slot[nums[0]]+unko_slot[nums[1]]+unko_slot[nums[2]]
        await ctx.send(mes)


    #mes1 = random.choice(unko_slot)
    #mes2 = random.choice(unko_slot)
    #mes3 = random.choice(unko_slot)
    #await ctx.send(mes1+mes2+mes3)
    #mes1 = random.choice(unko_slot)
    #mes2 = random.choice(unko_slot)
    #mes3 = random.choice(unko_slot)
    #await ctx.send(mes1+mes2+mes3)
    #mes1 = random.choice(unko_slot)
    #mes2 = random.choice(unko_slot)
    #mes3 = random.choice(unko_slot)
    #await ctx.send(mes1+mes2+mes3)

@bot.command(aliases=['でかスロット','でかすろっと','デカスロット','すろっと','スロット'])
async def com_deka_slot(ctx):
    global unko_slot
    global unko_slot2
    global custom_slot
    global kakuritu

    dice = list()
    for x in range(1,8):
        for y in range(1,8):
            for z in range(1,8):
                for xx in range(1,8):
                    for yy in range(1,8):
                        dice.append(10000*x+1000*y+100*z+10*xx+yy)
    #print(dice)

    #print(kakuritu)
    all_kaku = 1.000 - (kakuritu*7)
    #print(all_kaku)

    prob = list()
    for i in range(16807):
        if dice[i] == 11111:
            prob.append(kakuritu)
        elif dice[i] == 22222:
            prob.append(kakuritu)
        elif dice[i] == 33333:
            prob.append(kakuritu)
        elif dice[i] == 44444:
            prob.append(kakuritu)
        elif dice[i] == 55555:
            prob.append(kakuritu)
        elif dice[i] == 66666:
            prob.append(kakuritu)
        elif dice[i] == 77777:
            prob.append(kakuritu)
        else:
            prob.append(all_kaku/16800)

    samples = np.random.choice(a=dice,size=5,p=prob)
    #print(samples)


    rand_int = random.randint(0,1)
    if len(custom_slot) > 0:
        rand_int = random.randint(0,2)

    if rand_int == 0:
        for item in samples:
            n = item
            num = map(int, str(n))
            nums = list(num)
            print(nums)
            mes = unko_slot[nums[0]]+unko_slot[nums[1]]+unko_slot[nums[2]]+unko_slot[nums[3]]+unko_slot[nums[4]]
            await ctx.send(mes)
    elif rand_int == 1:
        for item in samples:
            n = item
            num = map(int, str(n))
            nums = list(num)
            print(nums)
            mes = unko_slot2[nums[0]]+unko_slot2[nums[1]]+unko_slot2[nums[2]]+unko_slot2[nums[3]]+unko_slot2[nums[4]]
            await ctx.send(mes)
    else:
        for item in samples:
            n = item
            num = map(int, str(n))
            nums = list(num)
            print(nums)
            mes = custom_slot[nums[0]]+custom_slot[nums[1]]+custom_slot[nums[2]]+custom_slot[nums[3]]+custom_slot[nums[4]]
            await ctx.send(mes)


@bot.command(aliases=['すろっとかすたむ','スロットカスタム'])
async def com_slot_custom(ctx, *args):
    
    global custom_slot
    global kakuritu

    if len(args) != 7:
        await ctx.send('絵文字を7つ指定してください')
        return

    if len(args) == 7:
        custom_slot = []
        custom_slot.append('')
        for item in args:
            custom_slot.append(item)

    #dice = list()
    #for x in range(1,8):
    #    for y in range(1,8):
    #        for z in range(1,8):
    #            for xx in range(1,8):
    #                for yy in range(1,8):
    #                    dice.append(10000*x+1000*y+100*z+10*xx+yy)
    #print(dice)

    #print(kakuritu)
    #all_kaku = 1.000 - (kakuritu*7)
    #print(all_kaku)

    #prob = list()
    #for i in range(16807):
    #    if dice[i] == 11111:
    #        prob.append(kakuritu)
    #    elif dice[i] == 22222:
    #        prob.append(kakuritu)
    #    elif dice[i] == 33333:
    #        prob.append(kakuritu)
    #    elif dice[i] == 44444:
    #        prob.append(kakuritu)
    #    elif dice[i] == 55555:
    #        prob.append(kakuritu)
    #    elif dice[i] == 66666:
    #        prob.append(kakuritu)
    #    elif dice[i] == 77777:
    #        prob.append(kakuritu)
    #    else:
    #        prob.append(all_kaku/16800)

    #samples = np.random.choice(a=dice,size=5,p=prob)
    
    await ctx.send('カスタムスロットを設定したよ♪')

    #for item in samples:
    #    n = item
    #    num = map(int, str(n))
    #    nums = list(num)
    #    mes = custom_slot[nums[0]]+custom_slot[nums[1]]+custom_slot[nums[2]]+custom_slot[nums[3]]+custom_slot[nums[4]]
    #    await ctx.send(mes)


@bot.command(aliases=['でかすろっとかすたむ','デカスロットカスタム'])
async def com_deka_slot_custom(ctx, *args):
    
    if len(args) != 7:
        await ctx.send('絵文字を7つ指定してください')
        return

    custom_slot = []
    custom_slot.append('')
    for item in args:
        custom_slot.append(item)



    dice = list()
    for x in range(1,8):
        for y in range(1,8):
            for z in range(1,8):
                for xx in range(1,8):
                    for yy in range(1,8):
                        dice.append(10000*x+1000*y+100*z+10*xx+yy)
    #print(dice)

    prob = list()
    for i in range(16807):
        if dice[i] == 11111:
            prob.append(0.0045)
        elif dice[i] == 22222:
            prob.append(0.0045)
        elif dice[i] == 33333:
            prob.append(0.0045)
        elif dice[i] == 44444:
            prob.append(0.0045)
        elif dice[i] == 55555:
            prob.append(0.0045)
        elif dice[i] == 66666:
            prob.append(0.0045)
        elif dice[i] == 77777:
            prob.append(0.0045)
        else:
            prob.append(0.9685/16800)

    samples = np.random.choice(a=dice,size=5,p=prob)
    

    for item in samples:
        n = item
        num = map(int, str(n))
        nums = list(num)
        mes = custom_slot[nums[0]]+custom_slot[nums[1]]+custom_slot[nums[2]]+custom_slot[nums[3]]+custom_slot[nums[4]]
        await ctx.send(mes)


@bot.command(aliases=['かべ','カベ','壁'])
async def com_kabe(ctx):
    global unko_slot

    
    for i in range(12):
        slots = ""
        for j in range(12):
            slots += random.choice(unko_slot)
        await ctx.send(slots)
    
    




@bot.command(aliases=['おみくじ'])
async def com_omikuji(ctx):
    global unko_omikuji
    #print(unko_omikuji)
    today_year = datetime.now(JST).strftime('%Y')
    #print(today_year)
    mes = random.choice(unko_omikuji)
    await ctx.send(f"{ctx.author.mention}"+'はんの'+today_year+'年の運勢は…')
    await ctx.send(mes+' やで！')

    

@bot.command(aliases=['おはよう'])
async def com_ohayo(ctx):
    global unko_ohayo
    mes = random.choice(unko_ohayo)
    await ctx.send(f"{ctx.author.mention}"+' '+mes)


@bot.command(aliases=['お知らせ'])
async def com_osirase(ctx):
    await ctx.send('@everyone はいはいみんな '+f"{ctx.author.mention}"+' が言いたいことがあるらしいで、ちょっと静かにしたってな、はいどうぞ')


@bot.command(aliases=['りろーど','リロード'])
async def com_reload(ctx):
    await ctx.send('読み込むでー')
    await func_all_reload()
    await ctx.send('読み込みんだで！おおきに！')


async def func_all_reload():
    await func_get_unko_message_spreadsheet()
    await func_get_unko_nemui_spreadsheet()
    await func_get_unko_ohayo_spreadsheet()
    await func_get_unko_massa_spreadsheet()
    await func_get_unko_tabeyo_spreadsheet()
    await func_get_unko_schedule_spreadsheet()
    await func_get_unko_slot_spreadsheet()
    await func_get_unko_omikuji_spreadsheet()
    await func_get_unko_log_spreadsheet()


async def func_get_unko_message_localhost():
    global unko_messages
    unko_messages.clear()
    unko_messages.append(['end','うん','こ',1,'💩'])
    unko_messages.append(['find','うんこ','なに？',1,'💩'])
    unko_messages.append(['find','くそ','なんや？',1,'💢'])
    print(unko_messages)

async def func_get_unko_message_spreadsheet():
    global unko_messages
    unko_messages.clear()
    last_line = int(sheet_messages.cell(1,2).value)
    column_size = 5
    ranges = sheet_messages.range(3,1,last_line,column_size)
    for start in range(0, len(ranges), column_size):
        values = []
        for vcell in ranges[start : start + column_size]:
            values.append(vcell.value)
        unko_messages.append(values)

async def func_get_unko_nemui_spreadsheet():
    global unko_nemui
    unko_nemui.clear()
    last_line = int(sheet_uranai.cell(1,2).value)
    column_size = 3
    ranges = sheet_uranai.range(3,1,last_line,column_size)
    for start in range(0, len(ranges), column_size):
        values = []
        for vcell in ranges[start : start + column_size]:
            values.append(vcell.value)
        unko_nemui.append(values)

async def func_get_unko_ohayo_spreadsheet():
    global unko_ohayo
    unko_ohayo.clear()
    last_line = int(sheet_ohayou.cell(1,2).value)
    ranges = sheet_ohayou.range(3,1,last_line,1)
    for vcell in ranges:
        unko_ohayo.append(vcell.value)

async def func_get_unko_massa_spreadsheet():
    global unko_massa
    unko_massa.clear()
    last_line = int(sheet_massa.cell(1,2).value)
    ranges = sheet_massa.range(3,1,last_line,1)
    for vcell in ranges:
        unko_massa.append(vcell.value)

async def func_get_unko_tabeyo_spreadsheet():
    global unko_tabeyo
    unko_tabeyo.clear()
    last_line = int(sheet_tabeyo.cell(1,2).value)
    ranges = sheet_tabeyo.range(3,1,last_line,1)
    for vcell in ranges:
        unko_tabeyo.append(vcell.value)

async def func_get_unko_schedule_spreadsheet():
    global unko_schedule
    unko_schedule.clear()
    last_line = int(sheet_schedule.cell(1,2).value)
    column_size = 3
    ranges = sheet_schedule.range(3,1,last_line,column_size)
    for start in range(0, len(ranges), column_size):
        values = []
        for vcell in ranges[start : start + column_size]:
            values.append(vcell.value)
        unko_schedule.append(values)

async def func_get_unko_slot_spreadsheet():
    global unko_slot
    global kakuritu
    kakuritu = float(sheet_slot.cell(1,2).value)
    unko_slot.clear()
    last_line = 10
    ranges = sheet_slot.range(3,1,last_line,1)
    for vcell in ranges:
        unko_slot.append(vcell.value)
    global unko_slot2
    unko_slot2.clear()
    last_line = 10
    ranges = sheet_slot.range(3,2,last_line,2)
    for vcell in ranges:
        unko_slot2.append(vcell.value)

async def func_get_unko_omikuji_spreadsheet():
    global unko_omikuji
    unko_omikuji.clear()
    last_line = int(sheet_omikuji.cell(1,2).value)
    ranges = sheet_omikuji.range(3,1,last_line,1)
    for vcell in ranges:
        unko_omikuji.append(vcell.value)

async def func_get_unko_log_spreadsheet():
    global unko_log
    global is_log_check
    is_log_check = False
    unko_log.clear()
    last_line = int(sheet_log.cell(1,2).value)
    column_size = 3
    ranges = sheet_log.range(3,1,last_line,column_size)
    for start in range(0, len(ranges), column_size):
        values = []
        for vcell in ranges[start : start + column_size]:
            values.append(vcell.value)
        unko_log.append(values)
    is_log_check = True






# spreadsheet から設定を読み込む
bot.loop.create_task(func_get_unko_message_spreadsheet())
bot.loop.create_task(func_get_unko_nemui_spreadsheet())
bot.loop.create_task(func_get_unko_ohayo_spreadsheet())
bot.loop.create_task(func_get_unko_massa_spreadsheet())
bot.loop.create_task(func_get_unko_tabeyo_spreadsheet())
bot.loop.create_task(func_get_unko_schedule_spreadsheet())
bot.loop.create_task(func_get_unko_slot_spreadsheet())
bot.loop.create_task(func_get_unko_omikuji_spreadsheet())
bot.loop.create_task(func_get_unko_log_spreadsheet())

bot.run(token)



