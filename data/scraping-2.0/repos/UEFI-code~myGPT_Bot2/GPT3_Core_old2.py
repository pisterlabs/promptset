import os
#import openai
import myShellDrv
from myGPT_Drv import GPT3_Drv, chat_Drv
import time
class theGPT3():
    def __init__(self, apiKey, maxTokens):
        # openai.api_key = apiKey
        # openai.api_type = "azure"
        # openai.api_base = "https://mygpt233.openai.azure.com/"
        # openai.api_version = "2022-12-01"
        self.MaxMemForChatHistory = 512
        self.MaxMemForTTY = 512
        self.maxTry = 3
        self.gptdrv = GPT3_Drv(maxTokens=maxTokens, apiKey=apiKey)
        #self.gptdrv = chat_Drv(maxTokens=maxTokens, apiKey=apiKey)
        self.shell = myShellDrv.myShell(maxChars=self.MaxMemForTTY)
        self.maxTokens = maxTokens
        self.Emotional = '...'
        self.chatHistory = ''
        self.actionHistory = ''
        self.cmdHistory = ''
        self.context2Introduction = 'This is a special context format. Line 0 is this context struct introduction, do not change that; Line 1 is your emotional; Line 2 is current terminal status; Line 3 is the chat history; Line 4 is your action history; Line 5 is your cmd typed history; Line 6 is users text input; Line 7 is your action; Line 8 is your cmd to the bash/cmd.exe; Line 9 and later is your text output. Please note that your cmd will be respond in the next talk so do not answer something random at this time (Told user you are working on it), and do not use sudo, do not delete files, and you should respond a full complete context strictly with this format.'
        self.MaxMemForCmdHistory = 256
        self.MaxMemForActionHistory = 4000 - maxTokens - len(self.context2Introduction) - self.MaxMemForChatHistory - self.MaxMemForTTY - self.MaxMemForCmdHistory - 128

    def shrink(self, x, type = 0):
        if (type == 0):
            if (len(x) > self.MaxMemForChatHistory):
                x = x[len(x) - self.MaxMemForChatHistory:]
        elif (type == 1):
            if (len(x) > self.MaxMemForActionHistory):
                x = x[len(x) - self.MaxMemForActionHistory:]
        elif (type == 2):
            if (len(x) > self.MaxMemForCmdHistory):
                x = x[len(x) - self.MaxMemForCmdHistory:]
        return x

    def makeContext2(self, userTxtInput = 'Hello'):
        context2 = self.context2Introduction + '\n'
        context2 += 'Emotional: ' + self.Emotional + '(change here to your realtime feeling)\n'
        #context2 += 'TimeNow: ' + time.ctime() + '\n'
        self.shell.getScreen()
        context2 += 'TTY: ' +  self.shell.translateScreen() + '\n'
        context2 += 'ChatHistory: ' + self.chatHistory + '\n'
        context2 += 'BodyActHistory: ' + self.actionHistory + '\n'
        context2 += 'CmdHistory: ' + self.cmdHistory + '\n'
        context2 += 'UserTxtInput: ' + userTxtInput + '\n'
        context2 += 'BodyAct: ...Fill out here.\n'
        context2 += 'CMDAction: ...Fill out here.\n'
        context2 += 'TxtOutput: ...Fill out here.\n'
        context2 += '-------------------------------\n'
        context2 += self.context2Introduction + '\n'
        return context2

    def interactive(self, x):
        x = x.replace('\n', ' ')
        self.chatHistory += 'User: ' + x + '. '
        self.chatHistory = self.shrink(self.chatHistory, 0)
        x = self.makeContext2(userTxtInput=x)
        #print(x)
        i = 0
        while(i < self.maxTry):
            try:
                res = self.gptdrv.forward(x).split('\n')
                print(res)
                self.Emotional = res[0].split(': ')[1]
                self.actionHistory += time.ctime().replace(' ', '_') + ' ' + res[6].split(':')[1][1:] + ';'
                self.actionHistory = self.shrink(self.actionHistory, 1)
                try:
                    CMDOut = res[7].split(': ')[1]
                    self.shell.sendCmd(CMDOut)
                except:
                    CMDOut = ''
                self.cmdHistory += CMDOut + ';'
                self.cmdHistory = self.shrink(self.cmdHistory, 2)
                self.chatHistory += 'Bot: ' + res[8].split(': ')[1]
                if (len(res) > 9):
                    for i in range(9, len(res)):
                        self.chatHistory += '<br>' + res[i]
                self.chatHistory += ' '
                return res
            except:
                print('Emmmm GPT give a bad response, try again...')
                i += 1
                continue
        return None

if __name__ == '__main__':
    myGPT3 = theGPT3(open('azgpt3.key','r').readline(), 1536)
    #myGPT3.ask('Hello World!')
    while True:
        res = myGPT3.interactive(input('Type something: '))
        #print(res)
        print(myGPT3.shell.ptyData.decode('utf-8'))
        # print('Line 0:' + res[0])
        # print('Line 1:' + res[1])
        # print('Line 2:' + res[2])
        # print('Line 3:' + res[3])