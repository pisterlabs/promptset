import os
#import openai
import myShellDrv
from myGPT_Drv import GPT4_Drv
import time
class theGPT4():
    def __init__(self, apiKey, name='CuteGPT'):
        self.name = name
        self.gptdrv = GPT4_Drv(apiKey=apiKey)
        self.shell = myShellDrv.myShell(maxChars=2048)
        self.EmotionHistory = ''
        self.chatHistory = ''
        self.actionHistory = ''
        self.cmdHistory = ''
        self.context2Introduction = 'This is a special context format. Follow this format strictly. Line 0 is this context struct introduction, do not change that; Line 1 is the chat history, do not change that; Line 2 is the emotion history, do not change that; Line 3 is the body action history, do not change that; Line 4 is users text input, you cannot change that; Line 5 is your action, you can do anything; Line 6 is your text output, you can say anything. Please respond a full complete context strictly with this format.'
        self.MaxMemForCmdHistory = 2048
        self.MaxMemForActionHistory = 2048
        self.MaxMemForChatHistory = 2048
        self.MaxMemForEmotionHistory = 2048
        self.maxTry = 5

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
        elif (type == 3):
            if (len(x) > self.MaxMemForEmotionHistory):
                x = x[len(x) - self.MaxMemForEmotionHistory:]
        return x

    def makeContext2(self):
        context2 = self.context2Introduction + '\n'
        self.shell.getScreen()
        ttyContent = self.shell.translateScreen()
        context2 += 'TTY: ' +  ttyContent + '\n'
        context2 += 'ChatHistory: ' + self.chatHistory + '\n'
        context2 += 'EmotionHistory: ' + self.EmotionHistory + '\n'
        context2 += 'BodyActHistory: ' + self.actionHistory + '\n'
        context2 += 'CmdHistory: ' + self.cmdHistory + '\n'
        #context2 += 'UserTxtInput: ' + userTxtInput + '\n'
        context2 += 'Emotional: ...Fill out here in English.\n'
        context2 += 'BodyAct: ...Fill out here in English.\n'
        context2 += 'CMDAction: ...Fill out here in bash/cmd.\n'
        context2 += 'TxtOutput: ...Fill out here\n'
        context2 += '-------------------------------\n'
        context2 += self.context2Introduction + '\n'
        context2 += 'TTY: ' +  ttyContent + '\n'
        context2 += 'ChatHistory: ' + self.chatHistory + '\n'
        context2 += 'EmotionHistory: ' + self.EmotionHistory + '\n'
        context2 += 'BodyActHistory: ' + self.actionHistory + '\n'
        context2 += 'CmdHistory: ' + self.cmdHistory + '\n'
        #context2 += 'UserTxtInput: ' + userTxtInput + '\n'
        return context2

    def interactive(self, x):
        x = x.replace('\n', ' ')
        self.chatHistory += 'User: ' + x + '. '
        self.chatHistory = self.shrink(self.chatHistory, 0)
        x = self.makeContext2()
        #print(x)
        i = 0
        while(i < self.maxTry):
            try:
                res = self.gptdrv.forward(x).split('\n')
                print(res)
                Emotion = res[0].split(': ')[1]
                BodyAct = res[1].split(': ')[1]
                try:
                    CMDOut = res[2].split(': ')[1]
                    self.shell.sendCmd(CMDOut)
                except:
                    print('Failed to prase CMDOut, use empty string instead.')
                    CMDOut = ''
                TxtOutput = '\n'.join(res[3:])
                TxtOutput = TxtOutput.split(': ')[1]
                self.EmotionHistory += Emotion + ';'
                self.shrink(self.EmotionHistory, 3)
                self.actionHistory += time.ctime().replace(' ', '_') + ' ' + BodyAct + ';'
                self.shrink(self.actionHistory, 1)
                self.cmdHistory += CMDOut + ';'
                self.shrink(self.cmdHistory, 2)
                if '\n' in TxtOutput:
                    self.chatHistory += self.name + ': ' + TxtOutput.replace('\n', '<br>')
                else:
                    self.chatHistory += self.name + ': ' + TxtOutput + ' '
                return Emotion, BodyAct, CMDOut, TxtOutput

            except:
                print('Emmmm GPT give a bad response, try again...')
                i += 1
                time.sleep(10)
                continue
        return None

if __name__ == '__main__':
    myGPT4 = theGPT4(open('azgpt3.key','r').readline())
    #myGPT3.ask('Hello World!')
    monitor_tty = input('Enter the Monitor TTY:')
    if (monitor_tty.startswith('/dev/tty')):    
        monitor_tty = open(monitor_tty, 'w')
    else:
        monitor_tty = None
    while True:
        Emo, Act, CMD, Txt = myGPT4.interactive(input('Type something: '))
        print('Emotion: ' + Emo)
        print('BodyAct: ' + Act)
        print('CMD: ' + CMD)
        print('Txt: ' + Txt)
        if monitor_tty:
            # Clear the screen
            monitor_tty.write('\x1b[2J')
            monitor_tty.write(myGPT4.shell.ptyData.decode('utf-8'))
            monitor_tty.flush()
        else:
            print(myGPT4.shell.ptyData.decode('utf-8'))
        # print('Line 0:' + res[0])
        # print('Line 1:' + res[1])
        # print('Line 2:' + res[2])
        # print('Line 3:' + res[3])