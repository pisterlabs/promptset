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
from imutils.video import VideoStream
import face_recognition
import imutils
import pickle
import cv2
import argparse

# This part of the code is from https://www.tomshardware.com/how-to/raspberry-pi-facial-recognition
# and slightly editted to our needs
def obtain_name_person():
    # Determine faces from encodings.pickle file model created from train_model.py
    encodingsP = "encodings.pickle"

    # load the known faces and embeddings along with OpenCV's Haar
    # cascade for face detection
    try:
        data = pickle.loads(open(encodingsP, "rb").read())
    except:
        return None

    # initialize the video stream and allow the camera sensor to warm up
    # Set the ser to the followng
    # src = 0 : for the build in single web cam, could be your laptop webcam
    # src = 2 : I had to set it to 2 inorder to use the USB webcam attached to my laptop
    vs = VideoStream(src=0, framerate=10).start()
    # vs = VideoStream(usePiCamera=True).start()

    # start the FPS counter

    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        # Detect the fce boxes
        boxes = face_recognition.face_locations(frame)
        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(frame, boxes)

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            name = "Unknown"  # if face is not recognized, then print Unknown

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    cv2.destroyAllWindows()
                    vs.stop()
                    return name
                cv2.destroyAllWindows()
                vs.stop()

                return None


async def run_test(name, device_num, ip):
    # Check if we recognized a face
    if name is None:
        name = ""
    global remove
    # Personalize greeting
    print(f"Hey {name}, you can talk now with your AI Mental Health Robot.")
    # Connect to the VOSK-server
    with sd.RawInputStream(samplerate=48000, blocksize = 4000, device=device_num, dtype='int16',
                            channels=1, callback=callback) as device:
            async with websockets.connect(ip) as websocket:
                await websocket.send('{ "config" : { "sample_rate" : %d } }' % (device.samplerate))

                while True:
                    # Obtain data
                    data = await audio_queue.get()
                    await websocket.send(data)
                    response = await websocket.recv()
                    try:
                        # Remove text when we are speaking
                        if json.loads(response)["partial"] != "":
                            if (pygame.mixer.music.get_busy()):
                                remove = True
                            continue
                    except KeyError:
                        if remove:
                            remove = False
                            continue
                        # Obtain final text
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
    # Fine-tune the model
    messages.append({"role": "system", "content": "You are a friendly chatbot who acts like a psychologist. Answer in the best way possible to "
                                            "help the client. Do never, in any circumstances, say something that can do harm. The client is also"
                                            " under the supervision of his own psychologist, so you can always refer them to the psychologist"
                                                  "if it is too much."})



async def add_message(inputMsg):
    # Add a message from the user
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
    # Print / speak response from Happy
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
        except Exception:
            pass






async def main(device_num, ip):
    os.system("clear")
    name = obtain_name_person()
    await startup()
    await run_test(name, device_num, ip)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip", type=str, default="127.0.0.1", help="IP address of VOSK-server")
    parser.add_argument("-k", "--key", type=str, default="empty", help="OpenAI Key")
    parser.add_argument("-d", "--device", type=int, default=0, help="Device number")
    args = parser.parse_args()
    openai.api_key = args.key
    device_num = args.device
    ip = "ws://" + args.ip + ":2700"
    asyncio.run(main(device_num, ip))

