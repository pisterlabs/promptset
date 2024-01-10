import os
import time 
import pygame.mixer
from gtts import gTTS
import openai
import speech_recognition as sr
import pyautogui
import pytesseract
import dotenv

dotenv.load_dotenv()

api_key = os.environ['API_KEY'] # OpenAI API KEY from the .env file
lang = "en"                     # [en, el]
openai.api_key = api_key

user = ""
microphone = sr.Microphone()

pygame.mixer.init()

def play_audio(text):
    speech = gTTS(text=text, lang=lang, slow=False, tld="com.au")
    speech.save("output.mp3")
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)
    pygame.mixer.music.unload()

def create_note_file(note: str, file_path: str):
    with open(file_path, "a") as f:
        f.write(note + "\n")

def read_note(file_path: str):
    with open(file_path, "r") as f:
        return f.readlines()
    
def capture_screenshot(file_path: str):
    screenshot = pyautogui.screenshot()
    screenshot.save(file_path)

def check_file_exists(dir: str, file_name: str, extension: str):
    file_path = f"{dir}{file_name}{extension}"
    if os.path.exists(file_path):
        file_counter = 1
        while os.path.exists(file_path):
            new_file_name = file_name + str(file_counter) + extension
            file_path = os.path.expanduser(f"{dir}{new_file_name}")
            file_counter += 1
    return file_path

def print_message(message: str):
    print("="*(len(message) + 4))
    print("= " + message + " =")
    print("="*(len(message) + 4))

def get_audio():
    r = sr.Recognizer()
    with microphone as source:
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio)
            print_message(said)
            global user
            user = said
            file_name = ""

            if "note" in said: 
                print_message("Preparing to write take a note...")
                play_audio("What would you like to make a note for?")
                note_audio = r.listen(source)
                note = r.recognize_google(note_audio)
                print_message(f"Note: \n{note}")
                note_dir = os.path.expanduser("~/Desktop/")
                file_name = "note"
                extension = ".txt"
                file_path = os.path.expanduser(f"~/Desktop/{file_name}{extension}")
                new_file_path = check_file_exists(dir=note_dir, file_name=file_name, extension=extension)
                play_audio(f"Note Saved as {file_name} on your Desktop!")
                print_message(f"Note Saved as {file_name} on {new_file_path}")
                create_note_file(note, new_file_path)
                while True:
                    play_audio("Would you like to take another note?")
                    another_note_audio = r.listen(source)
                    response = r.recognize_google(another_note_audio)
                    if "yes" in response:
                        play_audio("Would you like to add to the existing note?")
                        note_audio = r.listen(source)
                        response = r.recognize_google(note_audio)
                        if "yes" in response:
                            play_audio("What would you like to take a note for?")
                            note_audio = r.listen(source)
                            note = r.recognize_google(note_audio)
                            create_note_file(note, file_path)
                            play_audio("The note was added and saved!")
                        else:
                            file_name = "note"
                            play_audio("What would you like to take a note for?")
                            note_audio = r.listen(source)
                            note = r.recognize_google(note_audio)
                            file_path = os.path.expanduser(f"~/Desktop/{file_name}{extension}")
                            new_file_path = check_file_exists(dir=note_dir, file_name=file_name, extension=extension)
                            play_audio(f"Note Saved as {file_name} on your Desktop!")
                            print_message(f"Note Saved as {file_name} on {new_file_path}")
                            create_note_file(note, new_file_path)
                    else:
                        print_message("Preparing to hear your instructions...")
                        break
            elif "Reed" in said:
                play_audio("What file should I read from?")
                file_name_prompt = r.listen(source)
                file_name = r.recognize_google(file_name_prompt)
                file_path = os.path.expanduser(f"~\Desktop\{file_name}.txt")
                file_lines = read_note(file_path)
                for line in file_lines:
                    clean_line = line.replace("\n", "")
                    play_audio(clean_line)
                play_audio(f"End of file {file_name}")
            elif "Please" in said: 
                new_string = said.replace("Please", "")
                print_message(new_string)
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": new_string}])
                text = completion["choices"][0]["message"]["content"]
                play_audio(text)
            elif "screenshot" in said:
                print("Taking a screenshot...")
                screenshot_dir = os.path.expanduser("~/Desktop/")
                file_name = "screenshot"
                extension = ".png"
                new_file_path = check_file_exists(dir=screenshot_dir, file_name=file_name, extension=extension)
                print_message(f"Screenshot saved at {new_file_path}")
                capture_screenshot(new_file_path)
                play_audio("Screenshot saved!")
            elif "go" in said:
                play_audio("Note Bot Enabled. What can I do for you? Remember say Please first for using ChatGPT!")
        except Exception as e:
            print("Exception:", str(e))

while True:
    if "stop" in user:
        break
    get_audio()