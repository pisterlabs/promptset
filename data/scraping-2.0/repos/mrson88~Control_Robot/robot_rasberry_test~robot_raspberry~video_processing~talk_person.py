import sys
import speech_recognition as sr
import pyttsx3
import openai
openai.api_key='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'
count_check_listen=0
class TalkPerson():
    def __init__(self):
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.voice_id = 'vietnam+f3'
        self.engine.setProperty('voice', self.voice_id)
        self.speech_rate = 120
        self.engine.setProperty('rate', self.speech_rate)
        self.count_listen=0
        self.check_listen=True
        # -----
        self.item_listen=[]
        self.recognizer = sr.Recognizer()

        # -----------
        if self.engine :
            self.engine.say('xin chào sơn. Tôi có thể giúp gì cho bạn')
            self.engine.runAndWait()

    
    def listen(self):
        global count_check_listen
        if self.check_listen and self.count_listen<3:
            #second=time.strftime('%S')
            text_listen=self.convert_speech_to_text()
            print('text=',text_listen)
            if text_listen!='':
                chat_reserve_listen=self.chat_to_chatGPT(str(text_listen))
                print(chat_reserve_listen)
                self.engine.say(chat_reserve_listen)
                self.item_listen.append(text_listen)
                self.item_listen.append(chat_reserve_listen)
                self.engine.runAndWait()
                
            else:
                self.count_listen+=1
                count_check_listen+=1
                print(self.count_listen)
                print(count_check_listen)
        #if self.count_listen>=2:
            #self.check_listen=False
            #count_check_listen=0


    def stateChanged(self, state):
        if (state == self.engine.runAndWait()):
            self.ui.pushButton_say.setEnabled(True)
            self.ui.pushButton_listen.setEnabled(True)

    def convert_speech_to_text(self):
        # Use the default microphone as the audio source
        with sr.Microphone() as source:
            text=''
            print("Listening...")
            # Adjust the energy threshold to account for ambient noise
            self.recognizer.adjust_for_ambient_noise(source)
            # Listen for speech and convert it to text
            audio = self.recognizer.listen(source,timeout=5,phrase_time_limit=5)
            try:
                text = self.recognizer.recognize_google(audio, language='vi-VN')

                print("Converted text:", text)
            except sr.UnknownValueError:
                print("Unable to recognize speech")
            except sr.RequestError as e:
                print("Request error:", e)
            return text

    def chat_to_chatGPT(self,chat):
        completion = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[
            {"role": "user", "content": chat}
        ])
        return completion.choices[0].message.content


