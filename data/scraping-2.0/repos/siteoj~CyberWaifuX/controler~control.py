import threading
import logging
import time
import random
import datetime
from pycqBot.cqCode import image, record
from waifu.Waifu import Waifu
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from pycqBot.cqHttpApi import cqBot
from newvits.vits import voice_vits
from vits.fanyi import fanyi
rcaon = False
rcareply = ''
def newthread_process(waifu: Waifu,bot: cqBot):
    global rcaon,rcareply,rca
    while True:
        current_datetime = datetime.datetime.now()
        hour = current_datetime.strftime("%H")
        hour=int(hour)
        if (hour<=23 and hour>=7) and not rcaon :
            rca=threading.Thread(target=random_countdown_ask,args=(waifu,))
            rca.start()
        if  rcareply != '':
            
            ans=fanyi(s=rcareply,appkey=waifu.appkey,apiid=waifu.apiid)
            text = ans
            print(text)
            path=voice_vits(text=text)
            # time.sleep(5)
            path = path.replace("b'",'')
            path = path.replace("'",'')
            print(path)
            bot.cqapi.send_private_msg(waifu.qq_number,rcareply)
            bot.cqapi.send_private_msg(waifu.qq_number,text)
            time.sleep(0.5)
            bot.cqapi.send_private_msg(waifu.qq_number,"%s" % record(file='file:///' + path))
            
            # message.sender.send_message("%s" % record(file='http://192.168.1.102/VITS/output.wav'))
            
            logging.info('发送语音，文件目录是'+path)
            logging.info('发送自动关心信息\n'+rcareply) 
            rcareply = ''
            # break
        time.sleep(5)
def random_countdown_ask(waifu: Waifu): 
    global rcaon,rcareply
    rcaon = True
    interval = random.randint(45,180)
    # interval = 0.1
    logging.info(f'启动关心倒计时，{interval}分钟后将执行')
    time.sleep(int(interval*60))
    prompt = f'距离上次对话已经过去{interval}分钟了，你很关心{waifu.username},你应该和他找找话题，而不是干等着，这些我能提供给你的信息，继续和他聊天吧！'
    rcareply = waifu.ask(text=prompt)
    rcaon=False
    return
    
# rca=threading.Thread(target=random_countdown_ask,)