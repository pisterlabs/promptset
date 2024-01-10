import reflex as rx
import openai
import os
from text_to_speech import pytts
from dotenv import load_dotenv
from task import exec_task
import speech_recognition as sr
from multiprocessing import Queue
import time
transcript_queue = Queue()

load_dotenv()
openai.api_key = os.getenv('API_KEY') 

class State(rx.State):
    question: str
    processing: bool = False
    is_active_microphone: bool = False
    chat_history: list[tuple[str, str]]
    
    active: bool = False #flag to know if the Yeti is active
    sleep: bool = False #flag to know if the Yeti is sleeping
    task_mode: bool = False #flag to know if the Yeti is in task mode
    path_audio: str
    text_mic_on: str = "Mic On..."
    text_mic_off: str = "Mic off"
    
    @rx.var
    def get_text_mic_on(self) -> str:
        return self.text_mic_on
    
    @rx.var
    def get_text_mic_off(self) -> str:
        return self.text_mic_off
    
    @rx.var
    def get_path_audio(self) -> str:
        return self.path_audio
    
    def toggle_microphone(self):
        self.is_active_microphone = not self.is_active_microphone

    def micToWeb(self):
        r, mic = sr.Recognizer(), sr.Microphone()
        with mic as source:
            print("Escuchando para web...")
            try:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
                text = r.recognize_google(audio, language='es-ES')
                transcript_queue.put(text)
                print("msg to web: ", text, end="\r\n")  
            except:
                transcript_queue.put("")
                
    async def answer(self):
        session_answer = openai.ChatCompletion.create(
            model=os.getenv('MODEL'),
            messages=[
                {
                    "role": "system",
                    "content": os.getenv('CONTENT_SYSTEM')
                },
                {
                    "role": "user",
                    "content": self.question
                }
            ],
            stop=None,
            temperature=0.7,
            stream=True,
        )
        
        answer1 = ""
        self.chat_history.append((self.question, answer1))
        self.processing = True
        self.question = ""
        yield

        for i in session_answer:
            if hasattr(i.choices[0].delta, "content"):
                answer1 += i.choices[0].delta.content
                self.chat_history[-1] = (self.chat_history[-1][0], answer1)
                yield

        pytts(answer1)
        self.processing = False


    async def active_microphone(self):
        self.toggle_microphone()
        from record_microphone import micToWeb
        text_to_micro = micToWeb()
        text = text_to_micro

        if(text_to_micro != ""):
            session = openai.ChatCompletion.create(
                model = os.getenv('MODEL'),
                messages=[
                    {
                        "role": "system",
                        "content": os.getenv('CONTENT_SYSTEM')
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                stop=None,
                temperature=0.7,
                stream=True,
            )
            answer = ""
            self.chat_history.append((text, answer))
            yield
            
            for i in session:
                if hasattr(i.choices[0].delta, "content"):
                    answer += i.choices[0].delta.content
                    self.chat_history[-1] = (self.chat_history[-1][0], answer)
                    yield
            pytts(answer)
            self.toggle_microphone()

    async def active_microphone_infinite(self):
        self.toggle_microphone()
        print(self.is_active_microphone)
        while self.is_active_microphone:
            self.micToWeb()
            transcript_result = transcript_queue.get()

            if not self.sleep and not self.active and transcript_result == "yeti":
                self.active = True #set the flag to active
                answer = ""
                self.chat_history.append((transcript_result, os.
                getenv('text_chat_default')))
                yield
                for i in os.getenv('text_chat_default'):
                    answer += i
                    print('write answer', answer, end="\r\n")
                    self.chat_history[-1] = (self.chat_history[-1][0], answer)
                    yield
                pytts(os.getenv('text_chat_default')) #say the text chat message default
                print("AI web: ", os.getenv('text_chat_default'), end="\r\n")
                continue

            if not self.task_mode and transcript_result == "modo tarea":
                self.chat_history.append((transcript_result, os.getenv('text_chat_mode_task')))
                yield
                for i in os.getenv('text_chat_default'):
                    answer += i
                    print('write answer', answer, end="\r\n")
                    self.chat_history[-1] = (self.chat_history[-1][0], answer)
                    yield
                pytts(os.getenv('text_chat_mode_task')) #say the text mode task message
                self.task_mode = True #set the flag to active task mode
                continue

            #condition to interact with the Yeti in task mode
            if self.active and self.task_mode:
                task = exec_task() #execute the task
                print(task) 
                if task == 'Done':
                    self.chat_history.append((transcript_result, 'Tarea completada'))
                    yield
                    for i in 'Tarea completada':
                        answer += i
                        print('write answer', answer, end="\r\n")
                        self.chat_history[-1] = (self.chat_history[-1][0], answer)
                        yield
                    pytts('Tarea completada')

                if task == 'Exit':
                    self.chat_history.append((transcript_result, os.getenv('text_chat_not_mode_task')))
                    yield
                    for i in os.getenv('text_chat_not_mode_task'):
                        answer += i
                        print('write answer', answer, end="\r\n")
                        self.chat_history[-1] = (self.chat_history[-1][0], answer)
                        yield
                    pytts(os.getenv('text_chat_not_mode_task')) #say the text not mode task message
                    self.task_mode = False
                continue

            #condition to reactivate the Yeti
            if self.sleep and transcript_result == "yeti":
                self.active = True
                self.sleep = False
                self.chat_history.append((transcript_result, os.getenv('text_chat_after_sleep')))
                yield
                for i in os.getenv('text_chat_after_sleep'):
                    answer += i
                    print('write answer', answer, end="\r\n")
                    self.chat_history[-1] = (self.chat_history[-1][0], answer)
                yield
                pytts(os.getenv('text_chat_after_sleep'))
                print("\nAI: ", os.getenv('text_chat_after_sleep'), end="\r\n")
                continue

            #condition to sleep the Yeti 
            if self.active and transcript_result == "yeti adios" or transcript_result == 'yeti adiós' or transcript_result == "yeti apagate" or transcript_result == "yeti apágate":
                self.chat_history.append((transcript_result, os.getenv('text_chat_before_sleep')))
                yield
                for i in os.getenv('text_chat_before_sleep'):
                    answer += i
                    print('write answer', answer, end="\r\n")
                    self.chat_history[-1] = (self.chat_history[-1][0], answer)
                yield
                pytts(os.getenv('text_chat_before_sleep'))
                print("\nAI: ", os.getenv('text_chat_before_sleep'), end="\r\n")
                self.active = False
                self.sleep = True
                continue

            if not self.task_mode and not self.sleep and self.active and transcript_result != "":
                session = openai.ChatCompletion.create(
                    model = os.getenv('MODEL'),
                    messages=[
                        {
                            "role": "system",
                            "content": os.getenv('CONTENT_SYSTEM')
                        },
                        {
                            "role": "user",
                            "content": transcript_result
                        }
                    ],
                    stop=None,
                    temperature=0.7,
                    stream=True,
                )
                answer = ""
                self.chat_history.append((transcript_result, answer))
                # self.question = True
                yield
                
                for i in session:
                    if hasattr(i.choices[0].delta, "content"):
                        answer += i.choices[0].delta.content
                        # print('[-1]',self.chat_history[-1])
                        # print('[-1][0]',self.chat_history[-1][0])
                        self.chat_history[-1] = (self.chat_history[-1][0], answer)
                        yield
                pytts(answer)
                continue

            