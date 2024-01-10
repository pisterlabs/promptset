import os
import openai
from dotenv import load_dotenv
from elevenlabs import set_api_key,generate, play,voices,stream, VoiceSettings
from elevenlabs.api import Voices
import sys
import pygame
import json
import random
import pygame

def get_random_sound_from_folder(folder,wait):
    pygame.mixer.init()
    sounds = os.listdir(folder)
    random_sound = random.randint(0,len(sounds)-1)
    pygame.mixer.music.load(os.path.join(folder,sounds[random_sound]))
    pygame.mixer.music.play()
    if wait:
        while pygame.mixer.music.get_busy():
            pass

def play_audio(given_content):
  SKELETON = True
  if SKELETON:
    all_personalities = json.load(open('personalities.json', 'r',encoding="utf-8"))
    random_personality = ""
    count = 0
    for key in all_personalities:
      count += 1
    random_number = random.randint(0,count-1)
    count = 0
    for key in all_personalities:
      if count == random_number:
        random_personality = key
        break
      count += 1
    selected_personality =  all_personalities[random_personality]
    selected_personality.append({
            "role": "user",
            "content": given_content
      })
    messages = selected_personality
    print(f"\n {random_personality}")
    print(messages)
  load_dotenv()
  openai.api_key = os.getenv("OPENAI_API_KEY")
  set_api_key(os.getenv("ELEVEN_LABS_API_KEY"))
  #text = input("prompt")
  get_random_sound_from_folder("starting",False)
  completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages
  )
  
  answer = completion.choices[0].message["content"]
  print(answer)

  voices = Voices.from_api()
  final_voice = ""
  voice_to_find = ""
  if SKELETON:
    voice_to_find = "Deckard"
  else:
    voice_to_find = "John"
  for voice in voices:
    if voice.name == voice_to_find:
      print(voice)
      final_voice = voice

  final_voice.settings=VoiceSettings(stability=0.3,similarity_boost=0.1)

  print(final_voice)

  audio = generate(
      text=answer,
      voice=final_voice,
      model='eleven_monolingual_v1',
      stream=True
      
  )
  get_random_sound_from_folder("ending",False)
  stream(audio)