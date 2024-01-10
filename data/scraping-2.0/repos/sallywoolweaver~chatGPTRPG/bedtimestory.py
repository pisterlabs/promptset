import requests
import pyttsx3
import speech_recognition as sr
from gtts import gTTS
from dalleimages import Dalle
import threading
import pygame
import os
from dotenv import load_dotenv
import openai

load_dotenv()
API_KEY = os.environ.get("API_KEY")
openai.api_key = API_KEY


def chatStory(prompt):
    
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are DinoSnore, a social robot for children that creates and reads them bedtime stories. They should be based on a prompt the child comes up with. Be creative and fun!"},
                {"role": "user", "content": prompt},
            ],
            #stop=None,
            temperature=0,
        )

        story = response['choices'][0]['message']['content']
        return story
        
def summarize_story(story):
    if story:
        response = requests.post(
        "https://api.openai.com/v1/engines/text-davinci-002/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "prompt": "Summarize in 15 words this story: " + story,
                "max_tokens": 1024,
                "temperature": 0.5,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }
        )
        response_json = response.json()
        choices = response_json.get("choices")
        if choices:
            return choices[0].get("text")
    
    return None

def save_story(story):
    with open("story.txt", "w") as file:
        file.write(story)

def dis_story(summarized_story):
    Dalle(summarized_story)

def speak_textPYTT(story):
    engine = pyttsx3.init()
    engine.say(story)
    engine.runAndWait()

def speak_text(story):
    tts = gTTS(text=story, lang='en', slow=False)
    tts.save("story.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load("story.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(1000)

if __name__ == "__main__":
    r = sr.Recognizer()
    mic = sr.Microphone()

    print("Please say the type of story you want (e.g., fantasy, post-apocalyptic, science fiction).")
    prompt = ""
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        prompt = r.recognize_google(audio)
        print("You said:", prompt)
        
        story_prompt = f"Generate a long bedtime story. Give the main character a name. Make it educational. It should be PG-13. It should be set in: {prompt}"
        #print(chatStory(prompt))
        story = chatStory(story_prompt)
        print(story)
        summarized_story = summarize_story(story)

   
        text_thread = threading.Thread(target=speak_text, args=(story,))
        text_thread.start()

        save_story(story)
        dis_story(summarized_story)
        text_thread.join()

    except sr.UnknownValueError:
        print("Sorry, I couldn't understand your speech.")
        exit()
