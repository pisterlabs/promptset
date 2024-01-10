import logging
import collections, queue, os, os.path
import numpy as np
#import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal
import websockets
import asyncio
import traceback
import transcribe
# import translate
# import punctuate

logging.basicConfig(level=20)

logger = logging.getLogger('websockets')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

import os
import openai
openai.api_key = os.environ.get("OPENAI_API_KEY")

def make_iter():
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()
    def put(*args):
        loop.call_soon_threadsafe(queue.put_nowait, args)
    return queue, put

def main(source_lang, target_lang):
    # folder = "C:/Program Files (x86)/Steam/steamapps/common/NeosVR/data/tmp/"
    folder = "/home/guillefix/.steam/steamapps/common/NeosVR/data/tmp/"
    # Start audio with VAD
    async def translator():
        async for file in transcribe.transcribe_tokenizer_folder(folder):
            # naming convention - ID2C00_voice_tmp_[guid].wav
            # if file is not None:
            # username = str(file).split("_voice_")[0]
            # text = transcribe.transcribe_tokenizer(file)
            try:
                text = transcribe.transcribe_google(file)
                # ehm, do stuff with text if u want
                completion = openai.Completion.create(engine="ada", prompt="I am a very clever chatbot. \n. You: "+text+"\nMe:")
                yield text+"\n"+"<color=blue>"+completion.choices[0].text.split("\n")[0]+"</color>"
            except Exception as e:
                print(e)
                yield ""
            # punctuated = punctuate.punctuate(text.lower())
            # print("Recognized: %s" % punctuated)
            # if text is None:
            #     yield "hi"
            # else:
            #     yield text
            # yield punctuated
            # yield "hi"
            # translation = translate.translate(punctuated, source_lang, target_lang)
            # translation = punctuated
            # print("Translation: %s" % translation)
            # yield translation
            # os.remove(vad_file)

    async def test_generator():
        while True:
            yield "hi"
            await asyncio.sleep(2)
    async def test_generator2():
        things = test_generator()
        async for thing in things:
            yield thing
    # #
    async def send_result(websocket, path):
        # while not websocket.open:
        #     await asyncio.sleep(0.1)
        print("HOOO")
        result = translator()
        # result = test_generator2()
        async for msg in result:
            try:
                if msg != "":
                    await websocket.send(msg)
                    # print("oooooo")
                print(msg)
                # await websocket.send("hi")
                # await asyncio.sleep(1)
            except Exception as e:
                print(e)
                traceback.print_exc()
                break
        # while True:
        #     try:
        #         msg = await result.__anext__()
        #         print(msg)
        #         # await websocket.send("hohohoho")
        #         await websocket.send(msg)
        #         # await asyncio.sleep(0.1)
        #     except Exception as e:
        #         print(e)
        #         traceback.print_exc()

    # async def test():
    #     result = translator()
    #     async for msg in result:
    #         print(msg)

    async def test2():
        frames = vad_audio.vad_collector()
        print(frames)
        async for frame in frames:
            if frame is not None:
                print(frame)
    # asyncio.get_event_loop().run_until_complete(test2())

    ##To connect with Neos websockets
    asyncio.get_event_loop().run_until_complete(websockets.serve(send_result, 'localhost', 8765))

    # asyncio.get_event_loop().run_until_complete(test())
    asyncio.get_event_loop().run_forever()
    # translator()
