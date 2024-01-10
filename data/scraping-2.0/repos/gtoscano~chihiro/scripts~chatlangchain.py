from openai import OpenAI
import json  
import pygame
client = OpenAI()


def play_mp3(path_to_mp3):
    # Initialize pygame mixer
    pygame.mixer.init()

    # Load the MP3 file
    pygame.mixer.music.load(path_to_mp3)

    # Play the music
    pygame.mixer.music.play()

    # Wait for the music to play before exiting
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Replace 'your_file.mp3' with the path to your mp3 file
while True:
    user_input = input("Please enter your phrase: ")
    response_format = { "type": "json_object" }
    model = "gpt-3.5-turbo-1106"
    response = client.chat.completions.create(
      model=model,
      response_format=response_format,
      messages=[
        {"role": "system", "content": "You are a helpful french translator that always translates from any language to french and give a JSON output on the 'robot' key."},
        {"role": "user", "content": user_input}
      ]

    )
    #{"role": "system", "content": "You are a helpful french teacher that always check texts for grammar errors and typos and explains the problems you encounter and give a JSON output on the 'robot' key."},
    #{"role": "system", "content": "You are a helpful person who always replies any conversation and usually adds a follow-up question and give a JSON output on the 'robot' key."},
    translate_json = response.choices[0].message.content
    translate = json.loads(translate_json)
    print(translate)
    
    print('The translation is: ', translate['robot'])
    
    response = client.audio.speech.create(
        model="tts-1",
        voice="shimmer",
        input=translate['robot'],
    )
    
    response.stream_to_file("output.mp3")

    play_mp3('output.mp3')
