# -*- coding:UTF-8 -*-
######ChatGPTBot#######
####Author:EdmundFU####
import openai
from botoy import GroupMsg,Action,S,Botoy
from botoy import decorators as deco
from botoy.collection import MsgTypes
from botoy.decorators import these_msgtypes,from_these_groups
from botoy.contrib import plugin_receiver
import ast
from urllib import parse
openai.api_key = '1233'     #在这里输入你的API_KEY
MyBotName = '@' + '在这里替换为你的Bot的昵称' + ' '  #在这里输入你的Bot的昵称
@plugin_receiver.group
@deco.ignore_botself
@from_these_groups(   )        #这里填入监听的群聊
@these_msgtypes(MsgTypes.AtMsg)
def main(ctx=GroupMsg):
    if MyBotName in ctx.Content.strip():
        if MyBotName in ctx.Content.strip():
            MsgPre = ast.literal_eval(ctx.Content)
            Msg = MsgPre.get('Content')
            print(Msg.replace(MyBotName,''))
            if Msg.find(MyBotName) == 0:
                if len(Msg.replace(MyBotName,''))  > 2:     #这里是为了防止有人简单回复你好浪费API免费额度，如果有需要可以自行修改最短长度
                    response = openai.Completion.create(
                    model="text-davinci-003",            #这里的text-davinci-003为ChatGPT，ChatGPT在OpenAI内属于使用费高昂，如有需要可以修改为其他模型
                    prompt=str(Msg.replace(MyBotName,'')),
                    temperature=0.7,            #这里为模型觉得自己说的话的可信程度，0-1.0，越高ChatGPT的创意度越高
                    max_tokens=600,             #这里是最大长度，token不为实际字数，需要比例转换
                    )
                    print(response)
                    AnswerPre = response.get('choices')[0].get('text').replace('\\x', '%').encode('utf-8').decode('utf-8')
                    Answer = parse.unquote(AnswerPre)
                    S.bind(ctx).text(Answer,'utf-8')
                else:
                    S.bind(ctx).text("已禁止简单问题",'utf-8')
                    Action(ctx.CurrentQQ).shutUserUp(
                                                groupID=ctx.FromGroupId,
                                                userid=ctx.FromUserId,
                                                ShutTime=1
                                                )
bot = Botoy(
    qq = 123123123,         #在这里输入你的Bot的QQ号
    #use_plugins = True
)
if __name__ == "__main__":
    bot.run()