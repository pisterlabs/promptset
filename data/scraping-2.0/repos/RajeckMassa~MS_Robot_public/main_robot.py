import os
import openai
import sounddevice as sd
import asyncio
import websockets
import json
import sys
from gtts import gTTS
import pygame
from io import BytesIO
import pygame._sdl2.audio as sdl2_audio
import argparse


openai.api_key = ""

async def run_test(device_num, ip):
    global remove
    print(f"Hey, you can now talk with your AI Mental Health Robot.")
    # Connect to server
    with sd.RawInputStream(samplerate=48000, blocksize = 4000, device=device_num, dtype='int16',
                            channels=1, callback=callback) as device:
            async with websockets.connect(ip) as websocket:
                await websocket.send('{ "config" : { "sample_rate" : %d } }' % (device.samplerate))

                while True:
                    # Obtain audio
                    data = await audio_queue.get()
                    await websocket.send(data)
                    response = await websocket.recv()
                    try:
                        if json.loads(response)["partial"] != "":
                            if (pygame.mixer.music.get_busy()):
                                remove = True
                            continue
                    except KeyError:
                        if (remove):
                            # Remove if we are not done with the speaking
                            remove = False
                            continue
                        finalText = json.loads(response)["text"]
                        if len(finalText) < 10:
                            continue
                        os.system('clear')
                        print("You said: ", finalText)
                        await add_message(finalText)
                        continue

                await websocket.send('{"eof" : 1}')
                print (await websocket.recv())

# Function from the VOSK-documentation
def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    loop.call_soon_threadsafe(audio_queue.put_nowait, bytes(indata))

async def startup():
    global loop
    global audio_queue
    global messages
    global remove
    remove = False
    pygame.init()
    pygame.mixer.init()
    devices = tuple(sdl2_audio.get_audio_device_names())
    pygame.mixer.init(devicename=devices[1])

    loop = asyncio.get_running_loop()
    audio_queue = asyncio.Queue()
    messages = []
    # Fine-tune model
    messages.append({"role": "system", "content": "You are a friendly chatbot who acts like a psychologist. Answer in the best way possible to "
                                            "help the client. Do never, in any circumstances, say something that can do harm. The client is also"
                                            " under the supervision of his own psychologist, so you can always refer them to the psychologist"
                                                  "if it is too much."})



async def add_message(inputMsg):
    msg = {"role": "user", "content": inputMsg}
    print("AI Mental Health Robot: ", end='')
    sys.stdout.flush()

    messages.append(msg)
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages=messages,
        stream=True
    )
    n_chunk = 0

    sentence = ""

    for chunk in response:
        try:
            txt = chunk.choices[0].delta.content
            if (txt != "." and txt != "?" and txt != "!" and txt != ","):
                sentence += txt
            else:
                txt = chunk.choices[0].delta.content
                print(txt, end='')
                sys.stdout.flush()
                mp3_fp = BytesIO()
                tts = gTTS(sentence, lang='en', tld="com", slow=False)
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0) 
                if (n_chunk == 0):
                    pygame.mixer.music.load(mp3_fp)
                    pygame.mixer.music.play(0)
                else:
                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(1)
                    pygame.mixer.music.load(mp3_fp)
                    pygame.mixer.music.play(0)
                n_chunk += 1
                sentence = ""
                continue

            print(txt, end='')
        except Exception as e:
            pass






async def main(device_num, ip):
    os.system("clear")
    await startup()
    await run_test(device_num, ip)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip", type=str, default="127.0.0.1", help="IP address of VOSK-server")
    parser.add_argument("-k", "--key", type=str, default="empty", help="OpenAI Key")
    parser.add_argument("-d", "--device", type=int, default=0, help="Device number")
    args = parser.parse_args()
    openai.api_key = args.key
    device_num = args.device
    ip = "ws://" + args.ip + ":2700"
    asyncio.run(main(device_num, ip))

