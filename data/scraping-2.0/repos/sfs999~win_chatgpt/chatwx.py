#-*- coding: GBK-*-
import time
from wxauto import *
import openai
import os
#代理端口
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

#https://platform.openai.com/overview
openai.api_key="your_key"



def chatretern(prompt,moudel_engine="gpt-3.5-turbo"):
    cmpletion=openai.ChatCompletion.create(
        model=moudel_engine,
        messages=[{"role":"user","content":prompt}]
    )
    return cmpletion


if __name__ == '__main__':

    who = '文件传输助手' # 设置聊天对象，微信群名
    nickname = 'chatgpt'  # 触发chatgpt回复的关键字
    speakList = ['帆'] #设置谁可以发言
    wx = WeChat()
    wx.ChatWith(who)
    print("开始监控win微信程序")
    while True:
        msgobject1 = wx.GetLastMessage
        speaker1, msgcontent, speakerid1 = msgobject1
        time.sleep(1)
        # 如果收到的消息包含 chatgpt 的昵称，并且发件人在聊天群中：
        if nickname in msgcontent and speaker1 in speakList:

            wx.SendMsg('已收到 %s 的问题：' % (speaker1) + msgcontent[7:])
            print("已收到",'%s' % (speaker1),"的问题")
            sccess = False
            while not sccess:
                try:
                    ai_response = chatretern(msgcontent[7:])
                    returnMessage="sumtoken:"+str(ai_response.usage.total_tokens)+"  "+ai_response.choices[0].message['content']
                    sccess = True
                except:
                    wx.SendMsg('error! retrying...')
                    time.sleep(1)

            wx.SendMsg('@%s' % (speaker1) + returnMessage)
            print("已回复",'%s' % (speaker1),"的问题")
            continue
