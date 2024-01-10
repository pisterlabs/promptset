"""All the necessary imports
    """
import json
import os
import re
import tkinter
from collections import deque
from contextlib import suppress
from pickle import load
from threading import Thread
import customtkinter
import librosa as lb
import numpy as np
import openai
import pafy
import pandas as pd
import pyttsx3
import soundfile as sf
import speech_recognition as sr
import vlc
from bs4 import BeautifulSoup as bs
from requests import get as G

from funcs import Work  # type: ignore


class Prop:
    """ Contains all the Properties
    """
    def __init__(self) -> None:
        Thread(target=self.index_files).start()
        self.Query_queue = deque(maxlen=10)
        self.media_controller = 1

        with suppress(Exception):
            os.mkdir('./new_voices')

        self.result=np.array([])
        self.new_samples = len(os.listdir('./new_voices/'))
        self.focused_emotion_labels = ['neutral','happy', 'sad', 'angry']
        # self.emotion_labels = {
        # '01':'neutral',
        # '02':'happy',
        # '03':'sad',
        # '04':'angry',
        # }
        self.engine = pyttsx3.init('sapi5')
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)
        openai.api_key = r"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"
        with open(r"mlp_98_2.h5",'rb') as model:
            self.emotion_model = load(model)
            model.close()
    
    def index_files(self,directory = r"C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs"):
        ind = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".lnk"):
                    file_path = os.path.join(root, file)
                    file_name = os.path.basename(file_path).lower().replace('.lnk', '')
                    file_info = {
                        'name': file_name,
                        'path': file_path,
                    }
                    ind.append(file_info)
        with open("FileIndex.json", 'w',encoding='utf-8') as file:
            json.dump(ind, file)

    def audio_features(self,*args):
        """Audio features extraction 1
        """
        with sf.SoundFile(args[0]) as audio_recording:
            audio = audio_recording.read(dtype="float32")
            sample_rate = audio_recording.samplerate
            stft=np.abs(lb.stft(audio))
            mfccs=np.mean(lb.feature.mfcc(y=audio, sr=sample_rate,n_mfcc=500).T, axis=0)
            self.result=np.hstack((self.result, mfccs))
            return self.audio_features2(stft,sample_rate,audio,self.result)
         
    def audio_features2(self,*args):
        """Audio features extraction 2
        """
        chroma=np.mean(lb.feature.chroma_stft(S=args[0], sr=args[1]).T,axis=0)
        self.result=np.hstack((args[3], chroma))
        mel=np.mean(lb.feature.melspectrogram(y=args[2], sr=args[1]).T,axis=0)
        self.result=np.hstack((self.result, mel))
        return np.array(self.result)

class Listen(Prop):
    """Listens the User's Command
    Args:
        Prop (basic): _description_
    """
    # global media_controller
    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()
        self.text = None
        self.emotion = None
        self.conversation_history = []
        self.current_query = ''
        super().__init__()

    def speak(self,audio):
        """Converts text to speech

        Args:
            audio (str): Text to speech 
        """
        print(f"Assistant : - {audio}")
        with suppress(KeyboardInterrupt):
            self.engine.say(audio)
            self.engine.runAndWait()
            
    def replies(self):
        """Generate a reply With ChatGPT API call

        Returns:
            str: Returns reply in string
        """
        self.Query_queue.append(self.text)
        self.text = '\n'.join(self.Query_queue)
        message = [{
                    'role':'system',
                    'content':'You are Personal Assistant./Your name is Jarvis, who responds in the style of Jarvis.',
                   },
                    {'role':'user',
                     'content': self.text,
                     }
                ]
        print("Processing Request")
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message,
        max_tokens=2000,
        n=1,
        temperature=0.5,
        stop=["'''"],
        timeout=10,
        )
        res = str(response['choices'][0]['message']['content']).strip() # type: ignore
        self.Query_queue.append(res)
        return res
    
    def audio_processing(self):
        """Processes the audio data
        """
        audio_data = self.audio_features('input_data.wav').reshape(1,-1)
        self.result=np.array([])
        e_motion = self.emotion_model.predict(audio_data)[0]
        self.emotion=self.focused_emotion_labels[e_motion]
        print('Emotion: --> ',self.emotion)
        os.rename(
            f'./new_voices/input_data{str(self.new_samples)}.wav',
            f'./new_voices/{str(self.new_samples)}-' + f'{e_motion}' + '.wav',
        )
        self.new_samples+=1

    def mic_listen(self):
        """Accesses the mic for listening the command

        Returns:
            str : The generated reply
        """
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                self._extracted_from_mic_listen(source)
                return #self.mic_listen(res)
        except sr.UnknownValueError:
            self.mic_listen()
        except sr.WaitTimeoutError:
            self.mic_listen()
        except sr.RequestError:
            self.mic_listen()
        
    def audio_save(self,audio):
        """Saves Audio as Wav format
        Args:
            audio (wav): Audio Data from microphone
        """
        with open("input_data.wav", "wb") as file:
            file.write(audio)
            file.close()
        with open(f"./new_voices/input_data{str(self.new_samples)}.wav", "wb") as file:
            file.write(audio)
            file.close()
    # Rename this here and in `mic_listen`
    def _extracted_from_mic_listen(self, source):
        print("Listening")
        audio = self.recognizer.listen(source, timeout=3)
        thrd,thread = Thread(target=self.audio_processing),Thread(target=self.audio_save,args=(audio.get_wav_data(),))
        thread.start()
        thrd.start()
        print("Recognizing")
        self.current_query = self.text = str(self.recognizer.recognize_google(audio, language = 'en-IN')).lower()
        print(self.text)
        thread.join()
        thrd.join()


    def investigate(self,query=""):
        """To check whether to perform a task or not

        Args:
            query (str, optional): The kind of work to be performed. Defaults to "".

        Returns:
            function: Function that perform work
        """
        queries = set(query.split(' '))
        db,count,func = pd.read_excel("./DB.xlsx").to_dict('records'), 0.0, ""
        for keywords in db:
            keywordss = set(keywords['keywords'].split(','))
            freq = len(keywordss.intersection(queries))/len(keywordss)
            if freq>count:
                count = freq
                funcc = keywords['func']
                if 'self' not in funcc:
                    func = func.replace(func, f"self.work.{funcc}('{self.current_query}')")
                else:
                    func = func.replace(func,f"{funcc}('{self.current_query}')")
        print(func)
        return func

class App(customtkinter.CTk,Listen,Prop):
    def __init__(self):
        # self.rpl = ''
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme('green')
        super().__init__()
        Listen.__init__(self)
        Prop.__init__(self)
        self.work = Work()
        self.vlc = vlc.Instance('--aout=directx')
        self.geometry("600x500")
        self.resizable(True,True)
        self.title("Assistant 1.0")
        self.add_widget()

    def add_widget(self):
        ''' add widgets to app '''
        self.frame = customtkinter.CTkFrame(master=self,corner_radius=15)
        self.frame.place_configure(relx=0.5,rely=0.5,relheight=0.99,relwidth=0.5,anchor=tkinter.CENTER)
        self.cmd = customtkinter.CTkEntry(master=self.frame,placeholder_text="Enter Your Query",border_color='blue',corner_radius=15)
        self.cmd.place_configure(relx=0.01,rely=0.94,relheight=0.05,relwidth=0.9)
        self.cmd.bind('<Return>',self.textbox_input)
        self.app_title = customtkinter.CTkLabel(master=self.frame,text='Assistant',anchor=tkinter.CENTER,font=('Algerian',24,),text_color='#3366ff')
        self.app_title.place_configure(relx=0.3,rely=0.04,relwidth=0.4,relheight=0.06)
        self.button = customtkinter.CTkButton(master=self.frame,text="⊛",corner_radius=5,hover=True,hover_color='green',command=self.activate_mic)
        self.button.place_configure(relx=0.92,rely=0.94,relheight=0.05,relwidth=0.07)
        self.textbox = customtkinter.CTkTextbox(master=self.frame, corner_radius=15,state='disabled')
        self.textbox.place_configure(relx=0.01,rely=0.1,relheight=0.8,relwidth=0.98)
        self.textbox.tag_config('User',justify='right',foreground='#0099ff')
        self.textbox.tag_config('Assistant',justify='left',foreground='#1aff1a')
        # self.textbox.configure(state='disabled')

    def activate_mic(self):
        self.mic_listen()
        self.update_textbox()

    def pause_song(self):
        pass

    def play_pause_action(self,*_):
        if self.plp == 1:
            self.player.pause()
            self.play_pause_btn.configure(text='||',hover_color='green')
            self.plp-=1
        else:
            self.player.play()
            self.play_pause_btn.configure(text='▶',hover_color='red')
            self.plp+=1

    def GetHandle(self):
        self.frame1 = customtkinter.CTkFrame(master=self,corner_radius=15)
        self.frame1.place_configure(relx=0.75,rely=0.05,relheight=0.25,relwidth=0.25)
        self.plp = 1
        self.play_pause_btn = customtkinter.CTkButton(master=self,corner_radius=15,text='▶',hover=True,hover_color='red',command=self.play_pause_action)
        self.play_pause_btn.place_configure(relx=0.85,rely=0.3,relwidth=0.07)

    # Getting frame ID
        return self.frame1.winfo_id()
    def stop_song(self,cmd='stop'):
        """Stops song playing in background
        """
        with suppress(Exception):
            self.player.stop()
            self.player.release()
            self.frame1.destroy()
            self.play_pause_btn.destroy()
            if '1' not in cmd:
                self.media_controller += 1

    def play_h1(self,query):
        url = G(
            f"https://www.google.co.in/search?q={query.replace(' ', '+')}+latest+lyrical+youtube+song&tbm=vid",
            timeout=10,
            ).content
        if (
            soup := bs(url, 'lxml').find(
                'a',
                attrs={
                    'href': re.compile(
                        r"[/]url[?]q=https://www.youtube.com/watch(.*?)[&]"
                    )
                },
            )
        ):
            return soup  

    def play_h2(self,soup):
        youtube_url = str(re.search(
        r"https://www.youtube.com/watch(.*?)[&]",
        str(soup['href']),  # type: ignore
        )[0].replace(r'%3F', '?').replace(r'%3D', '=')[:-1])
        info = pafy.new(youtube_url)
        return info.getbest()

    def play_songs(self,query):
        """For playing songs 
        """
        if self.media_controller==1:
            self.media_controller -=1
        else:
            self.stop_song('stop1')
        Media = self.vlc.media_new(self.play_h2(self.play_h1(query)).url) #type: ignore
        Media.get_mrl()
        self.player = self.vlc.media_player_new()
        self.player.set_media(Media)
        self.player.set_hwnd(self.GetHandle())
        self.player.play()


    def isWork(self):
        self.rpl = ''
        if fun := self.investigate(self.current_query or ''):
            res = eval(fun)
        else: 
            self.rpl = self.replies()
        Thread(target=self.speak,args=(self.rpl,)).start()
    
    def update_textbox(self):
        self.isWork()
        self.textbox.configure(state='normal')
        self.textbox.insert(tkinter.END,f"{self.current_query} <-- User\n",'User')
        self.textbox.insert(tkinter.END,f"Assistant --> {self.rpl}\n\n",'Assistant')
        self.textbox.configure(state='disabled')
        self.cmd.delete(0,tkinter.END)
    # add methods to app
    def textbox_input(self,*_):
        self.text = self.current_query = self.cmd.get().lower()
        print(self.current_query)
        self.update_textbox()

# class open_dire():





if __name__ == "__main__":
    # sentence = "open the folder Contacts from disc c"  # the spoken voice converted into a sentence and passed in this variable

    # def open_folder(sentence):
    #     # Disc name keywords
    #     disc_keywords = ["disc", "disk"]
        
    #     # Extracting disc name and folder name
    #     disc_name = None
    #     folder_name = None
    #     words = sentence.split()
    #     for i in range(len(words)):
    #         if words[i] in disc_keywords and i+1 < len(words):
    #             disc_name = words[i+1]
    #         elif words[i] == "folder" and i+1 < len(words):
    #             folder_name = words[i+1]
    #     return [disc_name,folder_name]


    # def search_directory_in_disk(directory_path, target_directory):
    #     for root, dirs, files in os.walk(directory_path):
    #         if target_directory in dirs:
    #             target_directory_path = os.path.join(root, target_directory)
    #             print(f"The directory '{target_directory}' is found at '{target_directory_path}'.")
    #             os.startfile(target_directory_path)
    #             return True
    #     return False




    # disk_path = open_folder(sentence)[0]+':/'  # Specify the disk path you want to traverse
    # target_directory = open_folder(sentence)[1]   # Specify the directory name to search and open

    # if os.path.isdir(disk_path):
    #     if not search_directory_in_disk(disk_path, target_directory):
    #         print(f"The directory '{target_directory}' is not found in the disk '{disk_path}'.")
    # else:
    #     print(f"The disk path '{disk_path}' is not valid.")

    app = App()
    app.mainloop()