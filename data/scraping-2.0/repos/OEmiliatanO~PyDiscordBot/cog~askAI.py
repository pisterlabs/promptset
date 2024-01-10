import openai
from discord import Client as DC_Client, Message
from discord.ext import commands
from collections import deque
from random import choice, random, randint
from opencc import OpenCC
from aiohttp import ClientSession, TCPConnector, ClientTimeout
import asyncio
from asyncio.exceptions import TimeoutError
from cog.utilFunc import *
import pandas as pd
from time import localtime, strftime
import numpy as np
from collections import defaultdict
from os.path import isfile

MEMOLEN = 8
READLEN = 20
THRESHOLD = 0.85

with open('./acc/aiKey.txt', 'r') as acc_file:
    k, o = acc_file.read().splitlines()
    openai.api_key = k
    openai.organization = o
    
with open('./acc/banList.txt', 'r') as acc_file:
    banList = [int(id) for id in acc_file.readlines()]

scoreArr = pd.read_csv('./acc/scoreArr.csv', index_col='uid', dtype=int)
# with open('./acc/aiSet_base.txt', 'r', encoding='utf-8') as set2_file:
#     setsys_base = set2_file.read()
#     # setsys = {'role': 'system', 'content': acc_data}
#     setsys = {'role': 'system', 'content': setsys_base}
    
def localRead(resetMem = False) -> None:
    with open('./acc/aiSet_extra.txt', 'r', encoding='utf-8') as set1_file:
        global setsys_extra, name2ID, id2name, chatMem, chatTok, dfDict
        setsys_tmp = set1_file.readlines()
        setsys_extra = []
        name2ID, id2name = {}, []
        for i in range(len(setsys_tmp)//2):
            id2name.append(setsys_tmp[2*i].split(maxsplit=1)[0])
            name2ID.update((alias, i) for alias in setsys_tmp[2*i].split())
            setsys_extra.append(setsys_tmp[2*i+1])
        if resetMem:
            chatMem = [deque(maxlen=MEMOLEN) for _ in range(len(setsys_extra))]
            chatTok = [0 for _ in range(len(setsys_extra))]
            dfDict = defaultdict(pd.DataFrame)
        print(name2ID)

def nameChk(s) -> tuple:
    for name in name2ID:
        if name in s: return name2ID[name], name
    return -1, ''

def injectCheck(val):
    return True if val > THRESHOLD and val < 0.999 else False

whatever = [
    "對不起，發生 429 - Too Many Requests ，所以不知道該怎麼回你 QQ",
    "對不起，發生 401 - Unauthorized ，所以不知道該怎麼回你 QQ",
    "對不起，發生 500 - The server had an error while processing request ，所以不知道該怎麼回你 QQ"
    "阿呀 腦袋融化了~",
] + '不知道喔 我也不知道 看情況 可能吧 嗯 隨便 都可以 喔 哈哈 笑死 真假 亂講 怎樣 所以 🤔'.split()
   
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai.api_key}",
    "OpenAI-Organization": openai.organization,
}
# "organization": openai.organization,

cc = OpenCC('s2twp')

async def embedding_v1(inputStr:str):
    url = "https://api.openai.com/v1/embeddings"
    inputStr = inputStr.replace("\n", " ")
    async def Embed_Result(session:ClientSession, inputStr, url=url, headers=headers):
        data = {
            "model": "text-embedding-ada-002",
            "input": inputStr,
        }
        async with session.post(url, headers=headers, json=data) as result:
            return await result.json()
    async def get_response():
        to, co = ClientTimeout(total=60), TCPConnector(ssl=False)
        async with ClientSession(connector=co, timeout=to) as session:
            return await Embed_Result(session, inputStr)
    response = await get_response()
    if 'error' in response:
        return embedVector(str(response['error']), np.zeros(1536))
    return embedVector(inputStr, np.array(response['data'][0]['embedding']))

async def aiaiv2(msgs:list, botid:int, tokens:int) -> replyDict:
    url = "https://api.openai.com/v1/chat/completions"
    async def Chat_Result(session:ClientSession, msgs, url=url, headers=headers):
        data = {
            "model": "gpt-3.5-turbo-0301",
            "messages": msgs,
            "max_tokens": min(tokens, 4096-chatTok[botid]),
            "temperature": 0.8,
            "frequency_penalty": 0.6,
            "presence_penalty": 0.6
        }
        async with session.post(url, headers=headers, json=data) as result:
            return await result.json()

    async def get_response():
        to, co = ClientTimeout(total=60), TCPConnector(ssl=False)
        async with ClientSession(connector=co, timeout=to) as session:
            return await Chat_Result(session, msgs)

    response = await get_response()
    if 'error' in response:
        # print(response)
        return replyDict(rol='error', msg=response['error'])
    chatTok[botid] = response['usage']['total_tokens']
    if chatTok[botid] > 3000:
        chatMem[botid].popleft()
        chatMem[botid].popleft()
        print(f"token warning:{response['usage']['total_tokens']}, popped last msg.")
    return replyDict(msg = response['choices'][0]['message']['content'])
    
class askAI(commands.Cog):
    __slots__ = ('bot')
    
    def __init__(self, bot: DC_Client):
        self.bot = bot
        self.ignore = 0.5
        # self.last_reply = replydict()
    
    @commands.Cog.listener()
    async def on_message(self, message:Message):
        user, text = message.author, message.content
        uid, userName = user.id, user.global_name
        n = min(len(text), READLEN)
        
        if uid == self.bot.user.id:
            return
        
        elif (aiInfo:=nameChk(text[:n])) != (-1, ''):
            aiNum, aiNam = aiInfo
            
            # logging 
            print(f'{wcformat(userName)}[{aiNam}]: {text}')
            # hehe
            if uid in banList:
                if random() < self.ignore:
                    if random() < 0.9:
                        async with message.channel.typing():
                            await asyncio.sleep(randint(5, 15))
                        await message.channel.send(choice(whatever))
                    print("已敷衍.")
                    return
                else:
                    print("嘖")
                    
            elif ('洗腦' in text[:n]):
                if devChk(uid):
                    chatMem[aiNum].clear()
                    return await message.channel.send(f'阿 {aiNam} 被洗腦了 🫠')
                else:
                    return await message.channel.send('客官不可以')
                
            elif ('人設' in text[:n]) and devChk(uid):
                if ('更新人設' in text[:n]):
                    msg = text
                    setsys_extra[aiNum] = msg[msg.find('更新人設')+4:]
                return await message.channel.send(setsys_extra[aiNum])
            
            elif ('-t' in text[:n]) and devChk(uid):
                return await message.channel.send(f'Total tokens: {chatTok[aiNum]}')
            
            elif ('-log' in text[:n]) and devChk(uid):
                tmp = sepLines((m['content'] for m in chatMem[aiNum]))
                return await message.channel.send(f'Loaded memory: {len(chatMem[aiNum])}\n{tmp}')
            
            elif ('-err' in text[:n]) and devChk(uid):
                prompt = replyDict('user'  , f'{userName} said {text}' ).asdict
                reply  = await aiaiv2([prompt], aiNum, 99999)
                reply2 = sepLines((f'{k}: {v}' for k, v in reply.content.items()))
                print(f'{aiNam}:\n{reply2}')
                return await message.channel.send(f'Debugging {aiNam}:\n{reply2}')
            
            try:
                # 特判 = =
                if aiNum == 5: userName = '嘎零'
                prompt = replyDict('user'  , f'{userName} said {text}')
                
                if not uid in dfDict:
                    dfDict[uid] = pd.DataFrame(columns=['text', 'vector'])
                    # check if file exists
                    if isfile(f'./embed/{uid}.csv') and isfile(f'embed/{uid}.npy'):
                        tmptext = pd.read_csv(f'./embed/{uid}.csv')
                        tmpvect = np.load    (f'./embed/{uid}.npy', allow_pickle=True)
                        for i in range(len(tmptext)):
                            dfDict[uid].loc[i] = (tmptext.loc[i]['text'], tmpvect[i])
                
                if multiChk(text, ['詳細', '繼續']):
                    tokens = 500
                elif multiChk(text, ['簡單', '摘要', '簡略']) or len(text) < READLEN:
                    tokens = 60
                else:
                    tokens = 150
                    
                async with message.channel.typing():
                    # skipping ai name
                    if len(text) > len(aiNam):
                        nidx = text.find(aiNam, 0, len(aiNam))
                        if nidx != -1:
                            text = text[nidx+len(aiNam):]
                        if text[0] == '，' or text[0] == ' ':
                            text = text[1:]
                    # print(text)
                    embed = await embedding_v1(text)
                assert embed.vector[0] != 0
                
                idxs, corrs = simRank(embed.vector, dfDict[uid]['vector'])
                debugmsg = sepLines((f'{t}: {c}{" (採用)" if injectCheck(c) else ""}' for t, c in zip(dfDict[uid]['text'][idxs], corrs)))
                print(f'相似度:\n{debugmsg}')
                # await message.channel.send(f'相似度:\n{debugmsg}')
                # store into memory
                if len(corrs) == 0 or corrs[0] < 0.98:
                    dfDict[uid].loc[len(dfDict[uid])] = embed.asdict
                
                # filter out using injectCheck
                itr = filter(lambda x: injectCheck(x[1]), zip(idxs, corrs))
                selectMsgs = sepLines((dfDict[uid]['text'][t] for t, _ in itr))
                # print(f'採用:\n{selectMsgs} len: {len(selectMsgs)}')
                setupmsg  = replyDict('system', setsys_extra[aiNum] + f'現在是{strftime("%Y-%m-%d %H:%M", localtime())}')
                async with message.channel.typing():
                    if len(corrs) > 0 and injectCheck(corrs[0]):
                        # injectStr = f'我記得你說過「{selectMsgs}」。'
                        selectMsgs = selectMsgs.replace("\n", ' ')
                        prompt = replyDict('user', f'{userName} said {selectMsgs}，{text}')
                        print(f'debug: {prompt.content}')
                        
                    reply = await aiaiv2([setupmsg.asdict, *chatMem[aiNum], prompt.asdict], aiNum, tokens)
                assert reply.role != 'error'
                
                reply2 = reply.content
                # await message.channel.send(f'{cc.convert(reply2.replace("JailBreak", aiNam))}')
                await message.channel.send(f'{cc.convert(reply2)}')
            except TimeoutError:
                print(f'[!] {aiNam} TimeoutError')
                await message.channel.send(f'阿呀 {aiNam} 腦袋融化了~ 🫠')
            except AssertionError:
                if embed.vector[0] == 0:
                    print(f'Embed error:\n{embed.text}')
                if reply.role == 'error':
                    reply2 = sepLines((f'{k}: {v}' for k, v in reply.content.items()))
                    print(f'Reply error:\n{aiNam}:\n{reply2}')
                
                await message.channel.send(f'{aiNam} 發生錯誤，請聯繫主人\n{reply2}') 
            else:
                chatMem[aiNum].append(prompt.asdict)
                chatMem[aiNum].append(reply.asdict)
                # for i in chatMem[aiNum]:
                #     print(type(i))
                if uid not in scoreArr.index:
                    scoreArr.loc[uid] = 0
                scoreArr.loc[uid].iloc[aiNum] += 1
                
    
    @commands.hybrid_command(name = 'scoreboard')
    async def _scoreboard(self, ctx):
        user = ctx.author
        uid, userName = user.id, user.display_name
        if uid not in scoreArr.index: 
            return await ctx.send(f'{userName} 尚未和AI們對話過')
        arr = scoreArr.loc[uid]
        m = arr.max()
        i = int(arr.idxmax())
        s = arr.sum()
        t = scoreArr.sum(axis=1).sort_values(ascending=False).head(5)
        sb = sepLines((f'{wcformat(self.bot.get_user(i).global_name)}: {v}'for i, v in zip(t.index, t.values)))
        await ctx.send(f'```{sb}```\n{userName}最常找{id2name[i]}互動 ({m} 次)，共對話 {s} 次')
    
    @commands.hybrid_command(name = 'localread')
    async def _cmdlocalRead(self, ctx):
        user = ctx.author
        if devChk(user.id):
            localRead()
            await ctx.send('AI 人設 讀檔更新完畢')
        else:
            await ctx.send('客官不可以')

    @commands.hybrid_command(name = 'listbot')
    async def _listbot(self, ctx):
        t = scoreArr.sum(axis=0).sort_values(ascending=False)
        s = scoreArr.sum().sum()
        l = sepLines(f'{wcformat(id2name[int(i)], w=8)}{v : <8}{ v/s :<2.3%}' for i, v in zip(t.index, t.values))
        await ctx.send(f'Bot List:\n```{l}```')
            
    @commands.command(name = 'bl')
    async def _blacklist(self, ctx, uid):
        user = ctx.author
        # hehe
        if user.id in banList:
            return
        try:
            uid = int(uid)
            if uid not in banList and devChk(user.id):
                banList.append(uid)
                with open('./acc/banList.txt', 'a') as bfile:
                    bfile.write(str(uid))
                print(f'Added to bList: {uid}')
            else:
                print(f'Already banned: {uid}')
        except:
            print(f'ban error: {uid}')
    
    @commands.command(name = 'ig')
    async def _ignore(self, ctx, num):
        user = ctx.author
        # hehe
        if user.id in banList or not devChk(user.id):
            return
        num = float(num)
        self.ignore = num
        print(f'忽略率： {num}')

async def setup(bot):
    localRead(True)
    await bot.add_cog(askAI(bot))

async def teardown(bot):
    print('ai saved')
    # print(scoreArr)
    scoreArr.to_csv('./acc/scoreArr.csv')
    for k in dfDict.keys():
        print(f'UID {k}: {len(dfDict[k])}')
        dfDict[k]['text'].to_csv(f'./embed/{k}.csv', index=False)
        np.save(f'./embed/{k}.npy', dfDict[k]['vector'].to_numpy())
    