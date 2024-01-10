import json
import os
import sys
from datetime import datetime
import openai
import requests
import urllib.parse

import urllib3
import Misc.MTFileUtils as MTFileUtils
import Misc.MTStringUtils as MTStringUtils
from Misc.MTConfigIni import GMTConfig
from urllib3.exceptions import InsecureRequestWarning

urllib3.disable_warnings(InsecureRequestWarning)

# 记录本地缓存及会话信息
class ChatContext:
    host = ''
    session = ''
    username = ''
    messages = []
    filepath = ''
    maxcount = 10

    def AddUserMessage(msgstr):
        onemsg = {"role": "user", "content": msgstr}
        ChatContext.messages.append(onemsg)
        MTFileUtils.AppendText(ChatContext.filepath, "===================================================================\n")
        MTFileUtils.AppendText(ChatContext.filepath, "UserSay:" + msgstr + "\n")
        MTFileUtils.AppendText(ChatContext.filepath, "-------------------------------------------------------------------\n")
        

    def AddRspMessage(role, msgstr):
        onemsg = {"role": role, "content": msgstr}
        ChatContext.messages.append(onemsg)
        MTFileUtils.AppendText(ChatContext.filepath, "ChatGPT["+role+"]:" + msgstr + "\n")
        MTFileUtils.AppendText(ChatContext.filepath, "-------------------------------------------------------------------\n")
        

    def LoopMessage():
        oldmsgcnt = len(ChatContext.messages)
        if oldmsgcnt > ChatContext.maxcount:
            #print("Now MessageNum:", oldmsgcnt)
            #print("Max MessageNum:", ChatContext.maxcount)
            newqueue = []
            for i in range(oldmsgcnt - ChatContext.maxcount, oldmsgcnt):
                newqueue.append(ChatContext.messages[i])
            pass
            ChatContext.messages = newqueue
        pass



    def GetSessionCount()->int:
        n = len(ChatContext.messages)
        return n//2


# 用于封装中转服消息
class BotMessage:
    error = ""
    key = ""
    role = ""
    content = ""
    
    def __init__(self, rspdata:bytes) -> None:
        jstr = rspdata.decode()
        try:
            rspmsg = json.loads(jstr)
        except Exception as e:
            print("BotMessage Exception:")
            print("-------------------------------------------------------------------")
            print(jstr)
            print("-------------------------------------------------------------------")
            return
        pass
        self.key = rspmsg.get('key')
        self.error = rspmsg.get('error')
        if self.error == None:
            self.role = rspmsg.get('role')
            self.content = rspmsg.get('content')
            if self.content == None : 
                self.content = ''
            else:
                self.content = MTStringUtils.Base64StringDecode(self.content)
            pass
            if self.key == '' or self.key == None: self.error = 'key return error'
            if self.role == '' or self.role == None: self.role = 'bot'
            
        pass
        pass




# 与OpenAI直接交互
def ChatAPI(usermsg)->str:
    ChatContext.AddUserMessage(usermsg)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=ChatContext.messages)
    rspmsg = response.choices[0].message
    ChatContext.AddRspMessage(rspmsg.role, rspmsg.content)
    ChatContext.LoopMessage()
    return rspmsg.content

# 与中转服交互
def ChatBot(usermsg)->str:
    ChatContext.AddUserMessage(usermsg)
    host = ChatContext.host
    key = ChatContext.session
    usr = MTStringUtils.Base64StringEncode(ChatContext.username)
    msg = MTStringUtils.Base64StringEncode(usermsg)

    params = urllib.parse.urlencode({'key': key, 'usr': usr, 'msg': msg})
    url = f'{host}/yousay?%s' % params
    try:
        rsp = requests.get(url, verify=False)
        rspmsg = BotMessage(rsp.content)
    except Exception as e:
        print("ChatBot Execption:\n",str(e))
        print("-------------------------------------------------------------------")    
        return ''
    pass

    ChatContext.AddRspMessage(rspmsg.role, rspmsg.content)
    ChatContext.LoopMessage()
    if rspmsg.error != None and rspmsg.error != "":
        print("Error: ", rspmsg.error)
        print(rsp.content.decode())
        print("-------------------------------------------------------------------")    
    pass
    ChatContext.session = rspmsg.key
    return rspmsg.content


# main
if __name__ == '__main__':
    cmdpath = sys.argv[0]
    basedir = os.path.dirname(cmdpath)
    if basedir == '':
        basedir = os.curdir
    pass

    curtime = datetime.now()
    timestr = curtime.strftime("%Y-%m-%d_%H-%M-%S")
    ChatContext.filepath = basedir + '/log/chat.' + timestr + ".log"
    MTFileUtils.MakeSureParentDir(ChatContext.filepath)

    GMTConfig.Load(basedir + "/ChatGPTConsole.ini")

    # host
    ChatContext.host = GMTConfig.GetItemValue("Default","host")
    if ChatContext.host == '': ChatContext.host = 'https://127.0.0.1:4540'
    
    # direct
    direct = GMTConfig.GetItemValue("Default","direct") == '1'

    # api_key
    openai.api_key = GMTConfig.GetItemValue("Default", "api_key")
    if openai.api_key == '':
        print('api_key is empty')
    pass

    while(ChatContext.username == None or ChatContext.username == ''):
        print("===================================================================")
        username = input("YourName：")
        username.strip()
        ChatContext.username = username
    pass

    while(True):
        print("===================================================================")
        usermsg = input("YouSay[" + ChatContext.username + "]：")
        print("-------------------------------------------------------------------")
        if direct:
            rspmsg = ChatAPI(usermsg)
        else:
            rspmsg = ChatBot(usermsg)
        pass
        print("ChatGPT：", rspmsg)
        print("-------------------------------------------------------------------")
