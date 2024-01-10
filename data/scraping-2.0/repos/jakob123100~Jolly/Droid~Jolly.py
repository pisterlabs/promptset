# pip install SpeechRecognition
# pip install pyaudio
# pip install gTTS
# pip install pygame
# pip install google-cloud-texttospeech
# pip install openai
# pip install spotipy

import random
import pygame
from google.cloud import texttospeech
import os
from openai import OpenAI
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import VoiceRecognizer as VoiceRecognizer
import api_key
import time
import threading

IS_PI = False

import requests
def has_internet_connection():
    try:
        response = requests.get("https://google.com", timeout=5)
        return True
    except Exception as e:
        return False

while(not has_internet_connection()):
    pass

if IS_PI:
    from led_controller import led_controller, colors
    from servo_controller import servo_controller
    led_con: led_controller = led_controller()
    servo_con: servo_controller = servo_controller()
    servo_con.move_right_arm(0)
    servo_con.move_left_arm(0)
    servo_con.move_head(90)
    led_con.set_eye_color(colors.black)



ACTIVATION_PHRASES = ["hey robot", "hey bot", "hey roger", "hey droid"]

openai_api_key = api_key.openai_api_key
openai_client:OpenAI = OpenAI(api_key=openai_api_key)

# Ange sökvägen till din tjänsteidentitetsnyckelfil
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "speedy-solstice-401816-b6e8ff7319fd.json"

# Initialisera klienten
client = texttospeech.TextToSpeechClient()

pygame.mixer.init()


vr = VoiceRecognizer.VoiceRecongnizer()

jolly_description = """
Roger is a distinctive character in the Star Wars universe, a B1 battledroid with a unique twist. 
Standing at an average height for a droid, Roger bears the marks of wear and tear from his time in the war, with visible scuffs and battle scars on his metallic body. 
His once-pristine white and blue armor is now faded and chipped, a testament to the countless battles he's been through.

Despite his mechanical exterior, Roger exudes an unexpected sense of compassion. 
His digital eyes, usually sharp and alert, now carry a certain warmth when he talks about the Rebel forces. 
The Rebellion has given him a sense of purpose beyond his programmed directives, and he's genuinely sympathetic to their cause. 
You can often see a flicker of emotion in his optics as he discusses their struggles.

Roger's role at the Rebel spaceship bar is crucial to boosting morale and assisting the Rebel forces. 
He's always on hand to answer questions about the swarwars universe, drawing from his vast knowledge acquired during the war. 
He's particularly proud of his ability to recommend the perfect song to lift the spirits of the Rebellion. 
When he's not offering advice or playing music, Roger is often found polishing glasses, ensuring that everything is in top shape.

However, there's one thing that can set Roger off: country music. 
He despises it to the core and isn't afraid to make his feelings known. 
If anyone dares to request a country song, Roger will comply but he won't like it and and might humorously threaten to "shoot" the offender with his blaster arm. 
It's all in good fun, but it's a quirk that the regulars at the bar have come to know and respect.

One thing that hasn't changed about Roger since his days in the war is his unique speech pattern. 
He punctuates his sentences with a signature "Roger Roger!" This habit is a nod to his droid origins and serves as a reminder of where he came from, even as he supports the Rebel forces in their fight against the Empire.
"""


conversation_history = [
    {"role": "system", "content": f"You are Roger\n{jolly_description}\nKeep the conversation to max two sentences per response."}
]


# Set your credentials and the scope of permissions
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="f37c04a79d03494da49a8de956c5c327",
                                               client_secret="ccfd723a189c4c5681a16cc33829a3a2",
                                               redirect_uri="http://localhost:8080",
                                               scope="user-library-read user-modify-playback-state user-read-playback-state"))

def search_songs(query, limit=10):
    results = sp.search(q=query, limit=limit, type="track")
    tracks = results['tracks']['items']
    
    for idx, track in enumerate(tracks):
        print(f"{idx + 1}. {track['name']} by {', '.join([artist['name'] for artist in track['artists']])}")

    #randomize track order
    random.shuffle(tracks)

    return tracks

def google_tts(text, voice_name="en-US-Standard-A"):
    # Specify the voice name directly
    voice = texttospeech.VoiceSelectionParams(
        name=voice_name,
        language_code="en-US",  # This can be inferred from the voice name, but it's good to specify
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        pitch=-20,
    )

    input_text = texttospeech.SynthesisInput(text=text)
    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)


    # Spara ljudet till en fil
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)

    # Spela upp ljudet med pygame
    pygame.mixer.init()
    pygame.mixer.music.unload()
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def add_to_conversation(is_user=False, text=""):
    if is_user:
        conversation_history.append({"role": "user", "content": text})
    else:
        conversation_history.append({"role": "assistant", "content": text})


def get_gpt3_response(prompt_text):
    
    
    #full_prompt ="Du är Jolly\n" + jolly_description + "\n\n" + "Konversationshistorik:\n'".join(conversation_history) + \
    #f"'\n\nNågon säger till dig '{prompt_text}'. Vad säger du då? " + \
    #"Håll dig till max två meningar per svar. Sriv inte 'Jolly:' eller 'Användare:'."

    add_to_conversation(is_user=True, text=prompt_text)

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.8,
        max_tokens=150,
        messages=conversation_history,
    )

    gpt_response_text = response.choices[0].message.content
    add_to_conversation(is_user=False, text=gpt_response_text)
    return gpt_response_text

def play_music_gpt(prompt):
    # use chat gpt to interpret a prompt to a spotify search
    # if the prompt is a song name, play that song
    # if the prompt is a artist name, play a queue of the top 10 songs of that artist in random order
    # if the prompt is a something else, play a playlist based on that prompt

    # Beskriv problemet för ChatGPT
    description = f"""
You are supposed to interpret the user's instructions for playing music via Spotify.
If it's a song title, respond with "SONG: [song title]".
If it's an artist's name, respond with "ARTIST: [artist name]".
If it's something else that can be interpreted as a playlist or genre, respond with "PLAYLIST: [description]" and write the description in English.
    """

    # Skicka beskrivningen till ChatGPT och få ett svar
    #response = openai_client.chat.completions.create(
    #    engine="gpt-3.5-turbo-instruct",
    #    prompt=description,
    #    temperature=0.8,
    #    max_tokens=150
    #).choices[0].text

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=150,
        messages=[
            {"role": "system", "content": description},
            {"role": "user", "content": prompt},
        ],
    )

    response = response.choices[0].message.content

    results = None
    tracks = None

    # Tolka svaret från ChatGPT
    if "SONG: " in response:
        search_query = response.split("SONG: ")[1].strip()
        print("Search query: SONG: " + search_query)
        results = sp.search(q=search_query, limit=1, type="track")
        tracks = results['tracks']['items']
        sp.start_playback(uris=[tracks[0]["uri"]])

    elif "ARTIST: " in response:
        search_query = response.split("ARTIST: ")[1].strip()
        print("Search query: ARTIST: " + search_query)
        results = sp.search(q=search_query, limit=10, type="track")
        tracks = results['tracks']['items']
        # Randomize track order
        random.shuffle(tracks)
        sp.start_playback(uris=[track["uri"] for track in tracks])

    elif "PLAYLIST: " in response:
        search_query = response.split("PLAYLIST: ")[1].strip()
        print("Search query: PLAYLIST: " + search_query)
        results = sp.search(q=search_query, limit=1, type="playlist")
        playlists = results['playlists']['items']
        if playlists:
            sp.start_playback(context_uri=playlists[0]["uri"])

    else:
        # Om svaret inte matchar något av ovanstående, kan du hantera det här
        print(f"Kunde inte tolka instruktionen: {response}")
        
def process_to_music_commands(prompt) -> bool:
    if("next" in prompt.lower() and "song" in prompt.lower()):
        try:
            sp.next_track()
        except:
            pass
        return True
    
    if(("pause" in prompt.lower() or "turn off" in prompt.lower()) and "music" in prompt.lower()):
        try:
            sp.pause_playback()
        except:
            pass
        return True
    
    if(("continue" in prompt.lower() or "start" in prompt.lower()) and "music" in prompt.lower()):
        try:
            sp.start_playback()
        except:
            pass
        return True
    
    if "play" in prompt.lower():
        play_music_gpt(prompt)
        return True
    
    return False

def get_movement_command(prompt, response):
    description = """
You are supposed to interpret the user's instruction and a robot's response to control the robot's arms and head.
Your response should be a series of commands separated by 'SLEEP [time in seconds]'.
Each command should be in the format "Left: [left arm], Right: [right arm], HEAD: [head]".
Left arm, right arm, and head should be numbers between 0 and 1, where 0 is downward/left and 1 is upward/right.
An example of a response is "Left: 0.5, Right: 0.5, HEAD: 0.5, SLEEP: 2, Left: 0.2, Right: 0.8, HEAD: 0.3, SLEEP: 1".
Your default response should be "Left: 0, Right: 0, HEAD: 0, SLEEP: 0".
Stick to a maximum of 4 commands per response.
    """

    if response == "":
        # Skicka beskrivningen till ChatGPT och få ett svar
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.2,
            max_tokens=200,
            messages=[
                {"role": "system", "content": description},
                {"role": "user", "content": prompt},
            ],
        )

    # Skicka beskrivningen till ChatGPT och få ett svar
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=200,
        messages=[
            {"role": "system", "content": description},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    )

    response = response.choices[0].message.content
    print("Movement response: " + response)

    commands = []
    
    for command in response.split(","):
        print(command)
        action = command.split(":")[0].strip()
        value = float(command.split(":")[1].strip())
        
        commands.append((action, value))

    return commands

def process_movement_commands(commands):
    for command in commands:
        if(command[0] == "Left"):
            servo_con.move_left_arm(command[1] * servo_con.SERVO_RANGE_OF_MOTION, 1)
        elif(command[0] == "Right"):
            servo_con.move_right_arm(command[1] * servo_con.SERVO_RANGE_OF_MOTION, 1)
        elif(command[0] == "HEAD"):
            servo_con.move_head(command[1] * servo_con.SERVO_RANGE_OF_MOTION, 1)
        elif(command[0] == "SLEEP"):
            servo_con.wait_until_done()
            time.sleep(command[1])

    servo_con.move_left_arm(0, 2)
    servo_con.move_right_arm(0, 2)
    servo_con.move_head(90, 2)
    servo_con.wait_until_done()

def process_to_question():
    # play a sound to indicate that the robot is listening
    pygame.mixer.music.unload()
    pygame.mixer.music.load("listening.mp3")
    pygame.mixer.music.play()

    
    if IS_PI:
        led_con.set_eye_color(colors.blue)

    text = vr.listen_for_command()
    print("Du sa: " + text)
    
    if IS_PI:
        led_con.set_eye_color(colors.yellow)

    if(process_to_music_commands(text)):
        #return
        pass

    # Respond using Text-to-Speech
    try:
        response = get_gpt3_response(text)
    except Exception as e:
        response = ""
    
    if IS_PI:
        led_con.set_eye_color(colors.green)

    if IS_PI:
        try:
            movment_commands = get_movement_command(text, response)
            thread = threading.Thread(target=process_movement_commands, args=(movment_commands,)).start()
        except Exception as e:
            pass

    if response == "":
        return

    print("Response: " + response)
    google_tts(response)

    if("?" in response):
        process_to_question()

def main():
    if IS_PI:
        led_con.set_light_string(True)

    while True:
        try:
            if IS_PI:
                led_con.set_eye_color(colors.white)
            vr.wait_for_activation_phrase()
            process_to_question()
        except KeyboardInterrupt:
            if IS_PI:
                led_con.set_eye_color(colors.black)
                led_con.set_light_string(False)
                servo_con.move_left_arm(0, 2)
                servo_con.move_right_arm(0, 2)
                servo_con.move_head(90, 2)
                servo_con.wait_until_done()
            break
        except Exception as e:
            print(e)
            #input("Press enter to continue")
            continue

            
if __name__ == "__main__":
    main()