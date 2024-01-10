# -*- coding:UTF-8 -*-
####AIFreeChatV1.0####
import openai
from botoy import GroupMsg,Action,S,Botoy
from botoy import decorators as deco
from botoy.collection import MsgTypes
from botoy.decorators import these_msgtypes,from_these_groups
from botoy.contrib import plugin_receiver
import os
import ast
from urllib import parse
openai.api_key = 'XXXXXXXXXXXXXXXXXXXX' #填入你的API_KEY
Name = "XXXX"   #你的机器人昵称
@plugin_receiver.group
@deco.ignore_botself
@from_these_groups(XXXXXXXXX) #你的QQ号
@these_msgtypes(MsgTypes.AtMsg)
def main(ctx=GroupMsg):
    if Name in ctx.Content.strip():
        if Name in ctx.Content.strip():
            MsgPre = ast.literal_eval(ctx.Content)
            Msg = MsgPre.get('Content')
            print(Msg.replace(Name + " ",''))
            if Msg.find(Name) == 0:
                promptlist = []
                if len(Msg.replace(Name + " ",''))  > 2:
                    with open("Chat.txt","a") as w:
                        w.write("\nuserprompt:")
                        w.write(Msg.replace(Name + " ",''))
                        w.close()
                    with open("Chat.txt","r") as f:
                        for line in f.readlines():
                            line = line.strip('\n')
                            if 'systemprompt' in line:
                                promptdic = {}
                                promptdic.update({"role":"system"})
                                promptdic.update({"content":line.replace("systemprompt:","")})
                                print(promptdic)
                                promptlist.append(promptdic)
                                print("1")
                            if 'userprompt' in line:
                                promptdic = {}
                                promptdic.update({"role":"user"})
                                promptdic.update({"content":line.replace(Name + " ",'').replace("userprompt:","")})
                                promptlist.append(promptdic)
                                print("2")
                            if 'assistantprompt' in line:
                                promptdic = {}
                                promptdic.update({"role":"assistant"})
                                promptdic.update({"content":line.replace("assistantprompt:","")})
                                promptlist.append(promptdic)
                                print("3")
                        f.close()
                    print(promptlist)
                    try:
                        response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=promptlist,
                        temperature=0.7,
                        max_tokens=800
                        )
                        print(response)
                        AnswerPre = response.get('choices')[0].get('message').get('content').replace('\\x', '%').encode('utf-8').decode('utf-8')
                        Answer = parse.unquote(AnswerPre)
                        S.bind(ctx).text(Answer,True)
                        with open("Chat.txt","a") as a:
                            a.write("\nassistantprompt:")
                            a.write(Answer.replace("\n",""))
                            a.close()
                    except openai.error.APIConnectionError:
                        S.bind(ctx).text("网络丢包，请重试",'utf-8')
                    except openai.error.InvalidRequestError:
                        S.bind(ctx).text("对话缓存已满，已自动清理，请重试",'utf-8')
                        os.system("python3 replace_onetime.py")
                else:
                    S.bind(ctx).text("已禁止简单问题",'utf-8')
                    Action(ctx.CurrentQQ).shutUserUp(
                                                groupID=ctx.FromGroupId,
                                                userid=ctx.FromUserId,
                                                ShutTime=2
                                                )
bot = Botoy(
    qq = XXXXXXXX, #你的QQ机器人号码
    #use_plugins = True
)
if __name__ == "__main__":
    bot.run()
