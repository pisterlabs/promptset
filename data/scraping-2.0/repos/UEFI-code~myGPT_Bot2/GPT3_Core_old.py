import os
#import openai
from myGPT_Drv import GPT3_Drv

class theGPT3():
    def __init__(self, apiKey, maxTokens):
        # openai.api_key = apiKey
        # openai.api_type = "azure"
        # openai.api_base = "https://mygpt233.openai.azure.com/"
        # openai.api_version = "2022-12-01"
        self.gptdrv = GPT3_Drv(maxTokens=maxTokens, apiKey=apiKey)
        self.maxTokens = maxTokens
        self.context = ''
        self.Emotional = '...'
        self.chatHistory = ''
        self.actionHistory = ''
        self.context2Introduction = 'This is a special context format. Line 0 is this context struct introduction, do not change that; Line 1 is your emotional, you can manage it yourself freedom; Line 2 is the chat history, do not change that; Line 3 is your action history (<br> means line break here but you can use normal \\n in your response), do not change that; Line 4 is users text input, you cannot change that; Line 5 is your action, you can do anything; Line 6 is your text output, you can say anything. Please respond a full complete context strictly with this format.'

    def contextSpace(self):
        if(len(self.context) > self.maxTokens - 100):
            exceedNum = len(self.context) - (self.maxTokens - 100)
            self.context = self.context[exceedNum:]

    def ask(self, x):
        if x[-2:] != '\n\n':
            if x[-1] == '\n':
                x += '\n'
            else:
                x += '\n\n'
        self.context += x
        self.contextSpace()
        # response = openai.Completion.create(model="text-davinci-003",prompt=self.context,temperature=0.7,max_tokens=self.maxTokens,top_p=1,frequency_penalty=0,presence_penalty=0)
        # res = response['choices'][0].text
        res = self.gptdrv.forward(self.context)
        try:
            while res[0] == '\n':
                res = res[1:]
            if(res[-2:] != '\n\n'):
                if res[-1] == '\n':
                    self.context += (res + '\n')
                else:
                    self.context += (res + '\n\n')
            else:
                self.context += res
        except:
            pass
        return res
    
    def makeContext2(self, userTxtInput = 'Hello'):
        context2 = self.context2Introduction + '\n'
        context2 += 'Emotional: ' + self.Emotional + '(change here to your realtime feeling)\n'
        context2 += 'ChatHistory: ' + self.chatHistory + '\n'
        context2 += 'ActionHistory: ' + self.actionHistory + '\n'
        context2 += 'UserTxtInput: ' + userTxtInput + '\n'
        context2 += 'Action: ...Fill out here.\n'
        context2 += 'TxtOutput: ...Fill out here.\n'
        context2 += '-------------------------------\n'
        context2 += self.context2Introduction + '\n'
        return context2

    def interactive(self, x):
        x = x.replace('\n', ' ')
        self.chatHistory += 'User: ' + x + '. '
        x = self.makeContext2(userTxtInput=x)
        #print(x + '###########################\n')
        #response = openai.Completion.create(engine="myGPT3_5",prompt=x,temperature=0.7,max_tokens=self.maxTokens,top_p=1,frequency_penalty=0,presence_penalty=0)
        # response = openai.Completion.create(engine="myGPT3",prompt=x,temperature=0.7,max_tokens=self.maxTokens,top_p=1,frequency_penalty=0,presence_penalty=0)
        #response = openai.Completion.create(engine="myGPT3_Curie",prompt=x,temperature=0.7,max_tokens=self.maxTokens,top_p=1,frequency_penalty=0,presence_penalty=0)
        #res = response['choices'][0].text.split('\n')
        res = self.gptdrv.forward(x).split('\n')
        #print(res)
        #print('###########################\n')
        self.Emotional = res[0].split(': ')[1]
        self.actionHistory += res[4].split(':')[1][1:] + ';'
        self.chatHistory += 'Bot: ' + res[5].split(': ')[1]
        if (len(res) > 6):
            for i in range(6, len(res)):
                self.chatHistory += '<br>' + res[i]
        self.chatHistory += ' '
        return res

if __name__ == '__main__':
    myGPT3 = theGPT3(open('azgpt3.key','r').readline(), 2048)
    #myGPT3.ask('Hello World!')
    while True:
        res = myGPT3.interactive(input('Type something: '))
        print(res)
        # print('Line 0:' + res[0])
        # print('Line 1:' + res[1])
        # print('Line 2:' + res[2])
        # print('Line 3:' + res[3])