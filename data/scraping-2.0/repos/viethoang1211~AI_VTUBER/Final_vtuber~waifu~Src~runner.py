from colorama import *
import openai, humanize, os, sys, time, threading, asyncio, signal, json, webbrowser,edge_tts,sounddevice as sd,soundfile as sf, asyncio,torch 
from rich.console import Console
import whisper


import gpt3

# In test mode
CHARACTERAI_TURN_OFF =os.environ.get('CHARACTERAI_TURN_OFF')
initialized_prompt=""
# while True:
#     prompt = input("👦 > ")
#     try:
#         # Remove the 'proxy' variable and the 'proxies' parameter if you don't want to use a proxy.
#         # proxy = "Your proxies IP"
#         resp = gpt3.Completion.create(prompt=prompt, chat=[])
#         print(f"🤖 > {str(resp['text'])}")
#     except Exception as e:
#         print(f"🤖 > {str(e)}")

# test_prompt = "you are my best friend, how would you greet me?"
# resp = gpt3.Completion.create(prompt=test_prompt, chat=[])
# print(f"🤖 > {str(resp['text'])}")

# initialized_prompt= "You are an anime cat girl, you are cute, adorable, respond in lovely manner, now you will respond to my message: "


# If user didn't rename example.env
if os.path.exists("example.env") and not os.path.exists(".env"):
    os.rename("example.env", ".env")

# Load settings from .env file
with open('.env') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split('=', 1)
        os.environ[key] = value


# Set OpenAPI Key
if os.environ.get("OPENAI_KEY") is None:
    print(Fore.RED + Style.BRIGHT + "You didn't provide an OpenAI API Key!" + Style.RESET_ALL + " Things will not work.")
else:
    openai.api_key = os.environ.get("OPENAI_KEY")

# Check virtual env
print(Style.BRIGHT + Fore.GREEN)
if os.environ.get('VIRTUAL_ENV'):
    # The VIRTUAL_ENV environment variable is set
    print('You are in a virtual environment:', os.environ['VIRTUAL_ENV'])
elif sys.base_prefix != sys.prefix:
    # sys.base_prefix and sys.prefix are different
    print('You are in a virtual environment:', sys.prefix)
else:
    # Not in a virtual environment
    print(Fore.RED + 'You are not in a virtual environment, we\'ll continue anyways')

print(Style.RESET_ALL)

def signal_handler(sig, frame):
    print('Exiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

import utils.dependencies # Checks variables
import utils.audio
import utils.hotkeys
import utils.transcriber
import utils.characterAi
import utils.vtube_studio
import utils.translator
import utils.speech
import utils.punctuation_fixer


voice = os.environ.get("VOICE") 

utils.dependencies.start_check(voice)

if voice == "elevenlabs":
    import utils.elevenlabs
elif voice == "voicevox":
    import utils.voicevox
    utils.voicevox.run_async()
elif voice == "edge":
    import utils.voice_const

from rich.markdown import Markdown

utils.speech.prepare()
utils.characterAi.run_async()
utils.speech.silero_tts("hello", "en", "v3_en", "en_21")

# wtf
if json.loads(os.environ.get("VTUBE_STUDIO_ENABLED", "False").lower()):
    utils.vtube_studio.run_async()

print(Fore.RESET + Style.BRIGHT + "Welcome back, to speak press " + 
      (", ".join([Fore.YELLOW + x + Fore.RESET for x in utils.hotkeys.KEYS]) + " at the same time." if len(utils.hotkeys.KEYS) > 1 else utils.hotkeys.KEYS[0]))

semaphore = threading.Semaphore(0)

console = Console()


# import model


# We need to wait for this to end until the next
# input.
async def character_replied(raw_message):
    print(f"{Style.DIM}raw message: {raw_message}")
    print(f"\r{Style.RESET_ALL + Style.BRIGHT + Fore.YELLOW}Character {Fore.RESET + Style.RESET_ALL}> ", end="")
    
    # fix
    voice_message = utils.punctuation_fixer.fix_stops(raw_message)

    # Sometimes causes issues.
    console.print(Markdown(raw_message))

    # print(raw_message)

    if voice == "elevenlabs":
        try:
            utils.elevenlabs.speak(voice_message)
        except Exception as e:
            audio_path = utils.speech.silero_tts(voice_message)
            utils.audio.play_wav(audio_path, utils.vtube_studio.set_audio_level)
    elif voice == "voicevox":
        if json.loads(os.environ.get("TRANSLATE_TO_JP", "False").lower()):
            message_jp = utils.translator.translate_to_jp(voice_message)
            print(f"{Style.NORMAL + Fore.RED}jp translation {Style.RESET_ALL}> {message_jp}")
            if json.loads(os.environ.get("TTS_JP", "False").lower()):
                utils.transcriber.speak_jp(message_jp)

        if json.loads(os.environ.get("TTS_EN", "False").lower()):
            audio_path = utils.speech.silero_tts(voice_message)
            utils.audio.play_wav(audio_path, utils.vtube_studio.set_audio_level)
    elif voice == "edge":
        
        OUTPUT_EDGE= "edge_output.mp3"
        communicate = edge_tts.Communicate(raw_message, utils.voice_const.VOICE_FE_EN)
        await communicate.save(OUTPUT_EDGE)
        utils.audio.play_mp3(OUTPUT_EDGE, utils.vtube_studio.set_audio_level)
        

    # Set mouth to resting point
    utils.vtube_studio.set_audio_level(0)

    semaphore.release()

utils.characterAi.reply_callback = character_replied


# Variable for youtube live

chat =""
chat_now=""
chat_prev=""
owner_name="nova"
blacklist= ["Currently","testing"]

# End

# function for youtube live
import re
import pytchat

def yt_livechat(video_id):
        global chat

        live = pytchat.create(video_id=video_id)
        while live.is_alive():
        # while True:
            try:
                for c in live.get().sync_items():
                    # Ignore chat from the streamer and Nightbot, change this if you want to include the streamer's chat
                    # if c.author.name in blacklist:
                    #     continue
                    # if not c.message.startswith("!") and c.message.startswith('#'):
                    if not c.message.startswith("!"):
                        # Remove emojis from the chat
                        chat_raw = re.sub(r':[^\s]+:', '', c.message)
                        chat_raw = chat_raw.replace('#', '')
                        # chat_author makes the chat look like this: "Nightbot: Hello". So the assistant can respond to the user's name
                        chat = 'reply to: \" ' + c.author.name + ': ' + chat_raw + ' \"'
                        print(chat)
                        
                    time.sleep(1)
            except Exception as e:
                print("Error receiving chat: {0}".format(e))

semaphore_yt= threading.Semaphore(1)

def preparation():
    global chat_now, chat, chat_prev
    while True:
        # If the assistant is not speaking, and the chat is not empty, and the chat is not the same as the previous chat
        # then the assistant will answer the chat
        semaphore_yt.acquire()
        chat_now = chat
        if chat_now != chat_prev:
        # if is_Speaking == False and chat_now != chat_prev:
            # Saving chat history
            # conversation.append({'role': 'user', 'content': chat_now})
            chat_prev = chat_now
            # openai_answer()
            message = chat_now
            utils.characterAi.send_message_to_process_via_websocket(message)
            semaphore.acquire()         

        time.sleep(1)
        semaphore_yt.release()
    
# end function for youtube live

# function for mode 3
import speech_recognition as sr
semaphore1 = threading.Semaphore(1)

speech=""
speech_now=""
speech_prev=""
HAS_OPENAI= os.environ.get('OPENAI_CHECK')

def listen_and_respond():
    # Create a recognizer object
    global speech , HAS_OPENAI
    r = sr.Recognizer()
    r.energy_threshold = 3000
    
    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Say something...")
        
        while True:
            # Listen for audio input
            print("waiting for input...")
            semaphore1.acquire()
            audio = r.listen(source,phrase_time_limit=7)
            
            print("recieved audio:")
            # try:
                # Use the recognizer to convert speech to text
                
            text = r.recognize_sphinx(audio)
            if len(text) <= 15:
                semaphore1.release()
                continue
            else:
                pass
            
            with open("temp.wav", "wb") as f:
                f.write(audio.get_wav_data())
            
            # print("You said: " + text)
            # print("I heard that.")
            
            # for openai key 

            if HAS_OPENAI:
                audio_file= open("temp.wav", "rb")
                trans = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file,
                    temperature=0.1,
                    language="en"
                )
            else:
            # for non openai key
                result = utils.transcriber.audio_model.transcribe("temp.wav", fp16=torch.cuda.is_available())
                trans = result['text'].strip()


            if len(trans['text']) <=15:
                semaphore1.release()
                continue
            else:
                pass
            
            # print("You: " + trans['text'])
            speech = trans['text']
            semaphore1.release()

def preparation_1():
    global chat_now, chat, chat_prev,speech, speech_now,speech_prev
    while True:
        # If the assistant is not speaking, and the chat is not empty, and the chat is not the same as the previous chat
        # then the assistant will answer the chat
        semaphore_yt.acquire()
        chat_now = chat
        speech_now=speech
        semaphore1.acquire()
        if speech_now!= speech_prev:
            speech_prev = speech_now
            # openai_answer()
            message = speech_now
            utils.characterAi.send_message_to_process_via_websocket(message)
            semaphore.acquire() 
        else: 
            if chat_now != chat_prev:
            # if is_Speaking == False and chat_now != chat_prev:
                # Saving chat history
                # conversation.append({'role': 'user', 'content': chat_now})
                chat_prev = chat_now
                # openai_answer()
                message = chat_now
                utils.characterAi.send_message_to_process_via_websocket(message)
                semaphore.acquire()         
        semaphore1.release()
        time.sleep(1)
        semaphore_yt.release()


# for drawing
from opengpt import OpenGPT
import requests
from PIL import Image
import io

# end function



if  __name__ == "__main__":

    if CHARACTERAI_TURN_OFF ==True:
        initialized_prompt= "You are an anime cat girl, you are cute, adorable, respond in lovely manner, now you will respond to my message: "
    else:
        initialized_prompt= ""
    mode = input("Mode (1-Mic, 2-Youtube Live, 3-Live and listen): ")

    if mode =="1":
        print("Hold right ctrl and right shift to record audio")
        # Main process loop
        while True: 

            print(Style.RESET_ALL + Fore.RESET, end="")

            print("You" + Fore.GREEN + Style.BRIGHT + " (mic) " + Fore.RESET + ">", end="", flush=True)

            # Wait for audio input
            utils.hotkeys.audio_input_await()

            print("\rYou" + Fore.GREEN + Style.BRIGHT + " (mic " + Fore.YELLOW + "[Recording]" + Fore.GREEN +") " + Fore.RESET + ">", end="", flush=True)

            audio_buffer = utils.audio.record()

            # We need to keep track of the length of this message
            # because in Python we have no way to clear an entire line, wtf.
            try:
                tanscribing_log = "\rYou" + Fore.GREEN + Style.BRIGHT + " (mic " + Fore.BLUE + "[Transcribing (" + str(humanize.naturalsize(os.path.getsize(audio_buffer))) + ")]" + Fore.GREEN +") " + Fore.RESET + "> "
                print(tanscribing_log, end="", flush=True)
                transcript = utils.transcriber.transcribe(audio_buffer)
            except Exception as e:
                print(Fore.RED + Style.BRIGHT + "Error: " + str(e))
                continue


            # Clear the last line.
            print('\r' + ' ' * len(tanscribing_log), end="")
            print("\rYou" + Fore.GREEN + Style.BRIGHT + " (mic) " + Fore.RESET + "> ", end="", flush=True)

            # print("gotta check here..."+"\n")
            # print(f"{transcript.strip()}")    
            
            if transcript is None:
                continue
            print(f"{transcript.strip()}")    

            words= transcript.strip()
            words= words.replace("?","")
            words= words.replace("!","")
            words = words.replace(".", "")
            words = words.lower()
            words = words.split()
            
            # print(words.index("open"))

            if any(word in ["open", "start"] for word in words):
                word_index = words.index("open") if "open" in words else words.index("start")
                app = words[word_index + 1]
                # print(app)

                if app == "youtube":
                    asyncio.run(character_replied("okie"))
                    semaphore.acquire()
                    webbrowser.open("https://www.youtube.com/")
                elif app == "facebook":
                    asyncio.run(character_replied("okie"))
                    semaphore.acquire()
                    webbrowser.open("https://www.facebook.com/")
                elif app == "github":
                    asyncio.run(character_replied("okie"))
                    semaphore.acquire()
                    webbrowser.open("https://www.github.com/")
                elif app== "code":
                    asyncio.run(character_replied("okie"))
                    semaphore.acquire()
                    os.startfile(r"C:\Users\ADMIN\AppData\Local\Programs\Microsoft VS Code\Code.exe")
                # implement whatever you want here
                else:
                    utils.characterAi.send_message_to_process_via_websocket(transcript)
                    semaphore.acquire()
                    
                        
            elif any(word in ["sing","cover"] for word in words):
                word_index = words.index("sing") if "sing" in words else words.index("cover")
                song = ' '.join(words[word_index+1:])
                if "count on me" in song:
                    asyncio.run(character_replied("okie"))
                    semaphore.acquire()
                    time.sleep(2)
                    utils.audio.play_wav('song/countonme_ariana.wav', utils.vtube_studio.set_audio_level)
                    utils.vtube_studio.set_audio_level(0)
                    pass
                elif "jingle bell" in song:
                    pass
                elif ("go to return" in song) and ("vietnamese" or "vietnam" in song):
                    asyncio.run(character_replied("Go to return"))
                    semaphore.acquire()
                    time.sleep(2)
                    utils.audio.play_wav('song/didetrove_ariana.wav', utils.vtube_studio.set_audio_level)
                    utils.vtube_studio.set_audio_level(0)
                elif "happy birthday" in song:
                    asyncio.run(character_replied("okie"))
                    semaphore.acquire()
                    time.sleep(2)
                    utils.audio.play_wav('song/hpbd_ariana.wav', utils.vtube_studio.set_audio_level)
                    utils.vtube_studio.set_audio_level(0)
                elif ( "counting the days away from you" in song or "counting the days without you" in song) and ("vietnamese" or "vietnam" in song) :
                    asyncio.run(character_replied("I love that Vietnamese song, here you are"))
                    semaphore.acquire()
                    time.sleep(2)
                    utils.audio.play_wav('song/demngayxaem_ariana.wav', utils.vtube_studio.set_audio_level)
                    utils.vtube_studio.set_audio_level(0)
                elif ("someone else's wife" in song or "wife of someone else" in song) and ("vietnamese" or "vietnam" in song) :
                    asyncio.run(character_replied("this song is a bit difficult to sing."))
                    semaphore.acquire()
                    time.sleep(2)
                    utils.audio.play_wav('song/vonguoita_miku.wav', utils.vtube_studio.set_audio_level)
                    utils.vtube_studio.set_audio_level(0)
                elif "sorry" in song and ("vietnamese" or "vietnam" in song):
                    asyncio.run(character_replied("sorry"))
                    semaphore.acquire()
                    time.sleep(2)
                    utils.audio.play_wav('song/xinloi_ariana.wav', utils.vtube_studio.set_audio_level)
                    utils.vtube_studio.set_audio_level(0)
                elif "heal the world" in song:
                    asyncio.run(character_replied("okie"))
                    semaphore.acquire()
                    time.sleep(2)
                    utils.audio.play_wav('song/healtheworld_ariana.wav', utils.vtube_studio.set_audio_level)
                    utils.vtube_studio.set_audio_level(0)
                elif "perfect" in song:
                    asyncio.run(character_replied("okie"))
                    semaphore.acquire()
                    time.sleep(2)
                    utils.audio.play_wav('song/perfect_rose.wav', utils.vtube_studio.set_audio_level)
                    utils.vtube_studio.set_audio_level(0)
                else:
                    utils.characterAi.send_message_to_process_via_websocket(transcript)
                    semaphore.acquire()

            elif any(word in ["rap"] for word in words):
                word_index = words.index("rap")
                song = words[word_index+1]
                if song =="god":
                    asyncio.run(character_replied("rap god"))
                    semaphore.acquire()
                    time.sleep(1)
                    utils.audio.play_wav('song/rapgod_yuka.wav', utils.vtube_studio.set_audio_level)
                    utils.vtube_studio.set_audio_level(0)
                else:
                    utils.characterAi.send_message_to_process_via_websocket(transcript)
                    semaphore.acquire()
            elif any(word in ["draw"] for word in words):
                # print("read")
                asyncio.run(character_replied("I'm drawing"))
                semaphore.acquire()

                word_index = words.index("draw")
                picture = ' '.join(words[word_index+1:])
                hotpot = OpenGPT(provider="hotpot", type="image", options={"style": "Hotpot Art 9"})
                tem_link= hotpot.Generate(picture).url
                response =requests.get(tem_link)
                if response.status_code==200:
                    asyncio.run(character_replied("Here you are"))
                    semaphore.acquire()
                    image = Image.open(io.BytesIO(response.content))
                    image.save("tem.png")
                    os.startfile("tem.png")
                else:
                    print("Error:", response.status_code)
            else:    
                utils.characterAi.send_message_to_process_via_websocket(transcript)
                semaphore.acquire()

            # After use delete recording.
            try:
                # This causes ``[WinError 32] The process cannot access the file because it is being used by another process`` sometimes.
                # I don't know why.
                os.remove(audio_buffer)
            except:
                pass

    elif mode =="2":
        live_id = input("Livestream ID: ")
        # Threading is used to capture livechat and answer the chat at the same time
        t = threading.Thread(target=preparation)
        t.start()
        yt_livechat(live_id)
    elif mode =="3":
        live_id = input("Livestream ID: ")
        t = threading.Thread(target=preparation_1)
        t.start()
        t2= threading.Thread(target=listen_and_respond)
        t2.start()
        yt_livechat(live_id)        
    else:
        print("Invalid mode")