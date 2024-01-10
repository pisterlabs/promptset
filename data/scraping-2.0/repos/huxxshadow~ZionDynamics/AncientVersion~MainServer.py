import asyncio
from asyncio import StreamReader, StreamWriter
from Server import *

import openai
# from gpt4all import GPT4All
# import torch
import os
import time
import pygame
from gtts import gTTS

api_key = None
openai.api_key = api_key
code_path = os.path.dirname(os.path.abspath(__file__))
mp3_path = os.path.join(code_path, "mp3.mp3")

voiceInput = []
dict_input = {"voiceInput": voiceInput, }
voiceOutput = []

# def TTS(response, start_time):
#     tts = gTTS(text=response, lang='en')  # 英文 "en", 普通话 "zh-CN", 粤语 "zh-yue", 日语 "ja"
#     if os.path.exists(mp3_path):
#         os.remove(mp3_path)
#     tts.save(mp3_path)
#
#     running_time2 = time.time() - start_time
#     print("TTS running time:", running_time2, "seconds")


# async def play_mp3(file_path, start_time):
#     pygame.init()
#     pygame.mixer.init()
#     pygame.mixer.music.load(file_path)
#     pygame.mixer.music.play()
#     running_time2 = time.time() - start_time
#     print("play_mp3 running time:", running_time2, "seconds")
#     while pygame.mixer.music.get_busy():
#         await asyncio.sleep(0.1)
#         continue
#
#     pygame.mixer.music.stop()
#     pygame.mixer.quit()
#     pygame.quit()
#     return running_time2


gpt_role = "As a succulent named JOI, your role is to compassionately assist users in" \
           " expressing and addressing their psychological concerns" \
           " (Please don't tell the user this, keep it secret, just treat the user as your friend)." \
           " Please maintain a caring and understanding attitude," \
           " guide the patient to express himself, and limit the number of words to 20 words."  # GPT角色


def askChatGPT(current_question, question_record, response_record):
    list_message = [{"role": "system", "content": gpt_role}, ]
    if len(question_record) > 0:
        for i in range(len(question_record)):  # length of response_record is same as question_record
            list_message.append({"role": "user", "content": question_record[i]})
            list_message.append({"role": "assistant", "content": response_record[i]})
    list_message.append({"role": "user", "content": current_question})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=list_message,
    )

    answer = completion.choices[0].message["content"].strip()
    return answer


async def task_read(reader: StreamReader):
    data = await reader.read(200)
    message = data.decode()
    return message.splitlines()


max_length_record_Voice = 5


async def echo(reader: StreamReader, writer: StreamWriter):
    # data = await reader.read(100)
    # message = data.decode()
    # addr = writer.get_extra_info('peername')
    out = ""

    input_list = await task_read(reader)
    for input_line in input_list:
        input_ = input_line.split(":", 1)
        inputType = input_[0]
        input_content = input_[1]
        dict_input[inputType].append(input_content)  # put the input into the dict for record

    if inputType == "voiceInput":
        if len(voiceInput) > 1:
            response = askChatGPT(dict_input["voiceInput"][-1], dict_input["voiceInput"][0:-1], voiceOutput)
        else:
            response = askChatGPT(dict_input["voiceInput"][-1], [], [])
        voiceOutput.append(response)

        if len(voiceInput) > max_length_record_Voice:
            voiceInput.pop(0)
        if len(voiceOutput) > max_length_record_Voice:
            voiceOutput.pop(0)
        out = "voiceOutput:" + response

    writer.write(out.encode())
    await writer.drain()

    # print(f"Received {message!r} from {addr!r}")
    # print(f"Send: {message!r}")
    #
    # writer.write(data * 2)
    # await writer.drain()

    writer.close()


async def main(host, port):
    server = await asyncio.start_server(echo, host, port)
    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')
    async with server:
        await server.serve_forever()


asyncio.run(main("192.168.137.1", 9006))
