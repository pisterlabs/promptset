# import speech_recognition as sr
# import tkinter as tk
# import openai
# import os
# from gtts import gTTS
# import pyttsx3
# import openai
# import os

# # Set up OpenAI API credentials
# openai.api_key = "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"

# window = tk.Tk()

# window.title("My Window")

# window.geometry('500x300')

# e1 = tk.Entry(window, show=None,font=("Arial",14))

# e1.pack()




# # Initialize the speech recognition engine

# recognizer = sr.Recognizer()




# # Initialize the text-to-speech engine

# tts_engine = pyttsx3.init()




# # Set the voice for speech output

# # You can uncomment and modify this line based on your preferred voice

# # voices = tts_engine.getProperty('voices')

# # tts_engine.setProperty('voice', voices[1].id)





# def recognize_speech():

#     with sr.Microphone() as source:

#         print('Listening...')

#         recognizer.adjust_for_ambient_noise(source)

#         audio = recognizer.listen(source)




#     try:

#         text = recognizer.recognize_google(audio)

#         print("You said: ", text)

#         return text

#     except sr.UnknownValueError:

#         print("Speech recognition could not understand audio")


# def process_input(input_text):

#     response = openai.Completion.create(

#         engine="text-davinci-003",

#         prompt=input_text,

#         max_tokens=50

#     )

#     return response.choices[0].text.strip()

# def synthesize_speech(output_text):

#     tts_engine.save_to_file(output_text, 'open_ai_voice.wav')

#     tts_engine.runAndWait()

#     os.system("mediaplayer open_ai_voice.wav")
# def search_open_ai():

#     search_query = e1.get()

#     response = openai.Completion.create(

#         engine="text-davinci-003",

#         prompt=search_query,

#         max_tokens=50

#     )

# result = response.choices[0].text.strip()

#     print("Bot: ", result)

#     synthesize_speech(result)

#     e1.delete(0, tk.END)

# def user_input():

#     search_button = tk.Button(window, text="Search OpenAi", command=search_open_ai)

#     search_button.pack()




# def voice_input():

#     speech_button = tk.Button(window, text="Speak", command=speak_input)

#     speech_button.pack()

#     def speak_input():

#     text = recognize_speech()

#     e1.insert(tk.END, text)




# user_input()

# voice_input()

# window.mainloop()