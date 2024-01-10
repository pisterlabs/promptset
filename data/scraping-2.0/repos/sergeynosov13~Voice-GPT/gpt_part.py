import openai
import speech_recognition
from gtts import gTTS
import time
from playsound import playsound
import os
import random

import customtkinter
from PIL import ImageTk, Image
import time

def speak_completion():
    with open("completion.txt", "r") as f:
        text = f.read()
        print(text)

    language = "ru"

    obj = gTTS(text=text, lang=language, slow=False)

    name = random.randint(0,100)
    name_formed = str(name) + ".mp3"
    print(name_formed)
    obj.save(name_formed)

    time.sleep(3)
    print(name_formed)
    time.sleep(3)

    playsound(name_formed)

    os.remove(name_formed)
    time.sleep(0.01)
    os.remove(name_formed)
    time.sleep(0.01)
    os.remove(name_formed)
    time.sleep(0.01)
    os.remove(name_formed)
    time.sleep(0.01)
    os.remove(name_formed)
    time.sleep(0.01)
    os.remove(name_formed)


# chatgpt search function

def record():
    record_and_recognize_audio()
def gpt_request():
    openai.api_key = "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"
    with open("voice.txt", "r") as f:
        request = f.read()
    model_engine = "text-davinci-003"
    prompt = request
    print("Спасибо за запрос, ожидайте, я думаю.")
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    with open("completion.txt", "w+") as f:
        f.write(completion.choices[0].text)
        f.close()

    print("Ответ сохранен в файл")
    speak_completion()




# speak function
def record_and_recognize_audio(*args: tuple):
    recognizer = speech_recognition.Recognizer()
    microphone = speech_recognition.Microphone()
    with microphone:
        recognized_data = ""

        recognizer.adjust_for_ambient_noise(microphone, duration=2)

        try:
            print("Слушаю вас... Просьба произносить слова чётко")
            audio = recognizer.listen(microphone, 10, 10)

        except speech_recognition.WaitTimeoutError:
            print("Пожалуйста, проверьте работоспособность вашего микрофона")
            return

        try:
            print("Распознаю вашу речь...")
            recognized_data = recognizer.recognize_google(audio, language="ru").lower()

        except speech_recognition.UnknownValueError:
            pass
            print("Error1")

        except speech_recognition.RequestError:
            print("Пожалуйста, проверьте подключения к интернету")
            print("Error2")
    print(recognized_data)
    print("Вы сказали: ", recognized_data)
    with open("voice.txt", "w+") as f:
        f.write(recognized_data)
        f.close()
        print(recognized_data)
    gpt_request()



# speaking function

window = customtkinter.CTk()
window.geometry("750x750")
window.title("Voice GPT")
window.resizable(True,True)

def choise_theme(choise):
    if choise == "dark-blue":
        with open("theme.txt", "w+") as f:
            f.write("dark-blue")
            speak_lbl.configure(text="Перезагрузите программу")
    elif choise == "green":

        with open("theme.txt", "w+") as f:
            f.write("green")
            speak_lbl.configure(text="Перезагрузите программу")
with open("theme.txt", "r") as f:
    a = f.read()
    customtkinter.set_default_color_theme(a)
    print(a)

def request():
    record_and_recognize_audio()

gen_btn = customtkinter.CTkButton(window, text="Запрос", text_color="black",command=request)
speak_lbl = customtkinter.CTkLabel(window, text="После того как программа зависнет начните говорить через 2-3 секунды", text_color="White")
frame = customtkinter.CTkFrame(window)
theme = customtkinter.CTkComboBox(frame, values=["dark-blue", "green"], command=choise_theme)
img = customtkinter.CTkImage(Image.open("voice-transformed.png"), size=(100,100))
img_lbl = customtkinter.CTkLabel(window, image=img, text="")


img_lbl.grid(row=3, column=1,padx=10,pady=10)
frame.grid(row=4, column=0, padx=20, pady=(200, 10), sticky="s")
speak_lbl.grid(row=2,column=1, padx=10,pady=10)
gen_btn.grid(row=1,column=1, padx=10, pady=10)
theme.grid(row=4,column=1,padx=0,pady=10)

window.mainloop()
