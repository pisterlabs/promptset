import openai
import pyttsx3


class chatBot():
    def __init__(self, apiKey):
        openai.api_key = apiKey       
        self.voice = pyttsx3.init()
        self.voice.setProperty('rate', 185)
        self.history = [] #This is the history for a model using the openai.Completion.create api
        self.messages = [] #This is the message history for models using the openai.ChatCompletion.create api
        self.maxRowsInHistory = 42
        self.maxPromtLength = 4096
    
        self.messages.append({'role': 'system', 'content': 'You are talking like a normal human being. You have a lot of emotions and you are very intelligent. You also like to code, and you are very helpful and kind.'})
    
    def ChatLegacy(self, message, printText=True, useSound=True):
        self.history.append('You: ' + message + '\n')
        
        #Old version of the API   
        response = openai.Completion.create(
                      model="text-davinci-003",
                      prompt= self.GetChatHistory(),
                      temperature=0.5,
                      max_tokens=150,
                      top_p=1.0,
                      frequency_penalty=0.5,
                      presence_penalty=0.5,
                      stop=["You:"]
                    )        
        responseText = response.choices[0].text.lstrip('Friend:\n\n').lstrip('Bot:')
        self.history.append('Friend:\n\n' + responseText + '\n')
        
        if printText:
            print(responseText)
        
        if useSound:
            self.voice.say(responseText)
            self.voice.runAndWait()
        
        #Prune history to avoid overflow
        self.PruneHistory()
        
        return responseText

    def Chat(self, message, printText=True, useSound=True):
        self.messages.append({'role': 'user', 'content': message})
        
        response = openai.ChatCompletion.create(
                      model="gpt-4-0314",
                      messages= self.messages,
                      temperature=0.5,
                      max_tokens=500,
                      top_p=1.0,
                      frequency_penalty=0.5,
                      presence_penalty=0.5,
                      stop=["You:"]
                    )     
        
        responseText = response.choices[0].message.content.lstrip('\n\n')
        self.messages.append(response.choices[0].message)
        
        if printText:
            print(responseText)
        
        if useSound:
            self.voice.say(responseText)
            self.voice.runAndWait()
        
        #Prune history to avoid overflow
        self.PruneMessages()
        
        return responseText

    def GetChatHistory(self):
        return ''.join(self.history)[-self.maxPromtLength:]

    def PruneHistory(self):
        while(len(self.history) > self.maxRowsInHistory):
            self.history.pop()

    def PruneMessages(self):
        while(len(self.messages) > self.maxRowsInHistory):
            self.messages.pop()

    
    def ListVoices(self, testVoices = True):
        currentVoiceId = self.voice.getProperty('voice')
        print(f'CurrentVoiceId: {currentVoiceId}')
        voices = self.voice.getProperty('voices')
        
        for voice in voices:
            print(f'voiceId: {voice.id}. Name: {voice.name}')
            self.voice.setProperty('voice', voice.id)
            if testVoices:
                self.voice.say('The quick brown fox jumped over the lazy dog.')
                self.voice.runAndWait()
                
        self.voice.setProperty('voice', currentVoiceId)
        return voices
        
    def SetVoice(self, voiceId):
        self.voice.setProperty('voice', voiceId)
    
    def TestVoice(self, message):
        self.voice.say(message)
        self.voice.runAndWait()
        
        
    
    def ClearChatLog(self):
        self.history.clear()
        self.chatLog = ''