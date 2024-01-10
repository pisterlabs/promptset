import asyncio
import io
import wave
from asyncio import StreamReader, StreamWriter
from config_ import filler, valid_data_fixed_len, record_filler_len, data_package_size
import logging

import soundfile

from Server import *
import logging
import openai
import socket
import threading
import struct
import array

# from gpt4all import GPT4All
# import torch
import os
import time
# import pygame
from gtts import gTTS
# import base64
from pydub import AudioSegment
import soundfile as sf

import azure.cognitiveservices.speech as speechsdk

azure_key = None
region = 'eastus'
subscription_key = azure_key

# from scipy.io.wavfile import read as wavread
# from scipy.io.wavfile import write as wavwrite

# logging.basicConfig(format='[%(name)s] %(levelname)s: %(message)s', level=logging.INFO)
# logger = logging.getLogger("服务器")

# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
speech_config = speechsdk.SpeechConfig(subscription=None, region='eastus')

# The language of the voice that speaks.
speech_config.speech_synthesis_voice_name = "zh-CN-XiaoxiaoNeural"
# speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config,audio_config=None)
# speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff44100Hz16BitMonoPcm)
speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio48Khz96KBitRateMonoMp3)
file_config = speechsdk.audio.AudioOutputConfig(filename="temp.mp3")



# api_key = "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"
# api_key = "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"
api_key = None
openai.api_key = api_key
code_path = os.path.dirname(os.path.abspath(__file__))
mp3_path = os.path.join(code_path, "temp.mp3")

voiceInput = []
humidityInput=[]
dict_input = {"voiceInput": voiceInput,"humidityInput":humidityInput}
voiceOutput = []

STRING_SPECIFIER = "2222"
WAV_SPECIFIER = "3333"
exit = False
global control_2,t1
control_2=0;
t1=time.time()

# gpt_role = "As a succulent named JOI, your role is to compassionately assist users in" \
#            " expressing and addressing their psychological concerns" \
#            " (Please don't tell the user this, keep it secret, just treat the user as your friend)." \
#            " Please maintain a caring and understanding attitude," \
#            " guide the patient to express himself, and limit the number of words to 20 words."  # GPT角色
gpt_role = "作为一名名叫 JOI 的多肉植物，你的职责是充满爱心地帮助用户表达和解决他们的心理问题（请不要告诉用户这些，只把用户当作你的朋友）。请保持关心和理解的态度，引导患者表达，字数控制在 20 个字以内。根据上述回答的情感,必须从以下列表里[生气,流汗,哭哭,眨眼,惊讶,微笑]只挑选一个可以概括的内容的字符串在回答末尾用中括号围起来加上"  # GPT角色

global expSignal
expSignal=""
def askChatGPT(current_question, question_record, response_record):
    list_message = [{"role": "system", "content": gpt_role}, ]
    if len(question_record) > 0:
        for i in range(len(question_record)):  # length of response_record is same as question_record
            list_message.append({"role": "user", "content": question_record[i]})
            list_message.append({"role": "assistant", "content": response_record[i]})
    list_message.append({"role": "user", "content": current_question+"根据回答的情感,必须从以下列表里[生气,流汗,哭哭,眨眼,惊讶,微笑]只挑选一个可以概括的内容的字符串在回答末尾用中括号围起来加上"})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=list_message,
    )

    answer = completion.choices[0].message["content"].strip()
    print(answer)
    return answer


def TTS(textResponse):
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)
    result = speech_synthesizer.speak_text_async(textResponse).get()
    # audio_data = result.audio_data
    # with wave.open(mp3_path, 'wb') as wave_file:
    #     wave_file.setnchannels(1)  # Set the number of channels (1 for mono, 2 for stereo)
    #     wave_file.setsampwidth(2)  # Set the sample width (2 bytes for 16-bit audio)
    #     wave_file.setframerate(16000)  # Set the sample rate (e.g., 16000 Hz)
    #     wave_file.writeframes(audio_data)



    # tts = gTTS(text=response, lang='zh-CN')  # 英文 "en", 普通话 "zh-CN", 粤语 "zh-yue", 日语 "ja"
    # if os.path.exists(mp3_path):
    #     os.remove(mp3_path)
    # tts.save(mp3_path)

def mp3_to_wav(mp3_path):
    sound = AudioSegment.from_mp3(mp3_path)
    print(sound.frame_rate)
    sound.export("temp.wav", format="wav")


def receiveMsg():
    totalData = bytes()
    while True:
        data = sock.recv(1024)
        totalData += data
        if len(data) < 1024:
            break
    return totalData


def getData():
    # specifier = str(receiveMsg(), encoding="utf-8")
    # print("this is the specifier: " + specifier)
    # if specifier == STRING_SPECIFIER:
    receivedStr = str(receiveMsg(), encoding="utf-8")
    print("this is the receive msg: " + receivedStr+"###")
    return receivedStr
    # elif specifier == WAV_SPECIFIER:
    #     data = receiveMsg()
    #     ww = wave.open('received.wav', 'wb')
    #     ww.writeframes(data)
    #     ww.close()



def sendString(msg):
    if len(msg)==4:
        if msg[0]=="[" and msg[3]=="]":
            print("发送"+msg[1:3])
            send=msg[1:3].encode("utf-8")
            print(len(send))
            sock.sendall(send)
        else:
            print("发送wrong")
            send = "nu".encode("utf-8")
            sock.sendall(send)
    # if len(msg) % 1024 == 0:
    else:
        print("发送null")
        send="nu".encode("utf-8")
        sock.sendall(send)

# def get_package_from_file(reader):
#     real_data = reader.read(valid_data_fixed_len)
#     if not real_data:
#         return None
#
#     # 获取读取到的数据长度，不足使用填充符填充
#     current_data_len = len(real_data)
#
#     # 获取填充数据
#     fill_len = valid_data_fixed_len - current_data_len
#     fill_data = (fill_len * filler).encode()
#
#     # record_data 用来记录数据包的信息
#     record_data = f"{fill_len:0>{record_filler_len}d}".encode()
#
#     send_data = real_data + fill_data + record_data
#     logger.debug(f"待发送数据包的 真实数据-填充-填充字符长度分别是: {len(real_data)} - {len(fill_data)} - {len(record_data)}")
#
#     # 将768个字节 utf8编码数据  转换1024个字节 base64编码数据
#     data_package = base64.b64encode(send_data)
#     return data_package


def sendWAV(songPath):
    # sock.sendall(bytes(WAV_SPECIFIER, encoding="utf-8"))

    print("sending wav file")
    #

    #
    # try:
    #     with open("temp.wav", "rb") as rf:
    #         while True:
    #             # 读取文件数据包
    #             data_package = get_package_from_file(rf)
    #             if not data_package:
    #                 logger.info("传输完成！！")
    #                 break
    #             sock.send(data_package)
    #             logger.debug(f"数据包为{data_package} 长度:{len(data_package)}")
    # except Exception:
    #     logger.warning("文件不存在！")



    with open(songPath, "rb") as wavfile:
        input_wav = wavfile.read()
    sock.sendall(int.to_bytes(len(input_wav), 4, byteorder="little"))
    time.sleep(0.05)
    sock.sendall(input_wav)
    print(len(input_wav))
    print("send wav file")
    # rate, data = wavread(io.BytesIO(input_wav))

    # sock.sendall(rate)
    # time.sleep(1)
    # sock.sendall(data)
    # time.sleep(1)
    # with open(songPath,"rb") as file:
    #     data=file.read()
    # sock.sendall(data)
    # data, samplerate = sf.read(songPath)
    # print(samplerate)
    # sock.sendall(pickle.dumps(samplerate))
    # time.sleep(1)
    # data_=pickle.dumps(data)
    # print(data_)
    # print(len(data_))
    # sock.sendall(data_)
    # time.sleep(1)



    # file = wave.open(songPath, 'rb')
    # songData = str(file.getframerate()) + " " + str(file.getnframes())
    # sock.sendall(songData.encode("utf-8"))
    # sock.sendall(str(file.getframerate()).encode("utf-8"))
    # time.sleep(1)
    # sock.sendall(str(file.getnframes()).encode("utf-8"))
    # time.sleep(1)
    # sock.sendall(file.readframes(file.getnframes()))
    # time.sleep(1)

    # sock.sendall(bytes(str(file.getnchannels()), encoding="utf-8"))
    # sock.sendall(bytes(str(file.getsampwidth()), encoding="utf-8"))
    # sock.sendall(int.to_bytes(file.getnchannels()))
    # sock.sendall(int.to_bytes(file.getsampwidth()))
    # sock.sendall(int.to_bytes(file.getnframes(),4,byteorder="little"))

def handleMsg(msg):
    input_list = msg.splitlines()
    out=""
    for input_line in input_list:
        input_ = input_line.split(":", 1)
        inputType = input_[0]
        input_content = input_[1]
        dict_input[inputType].append(input_content)

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
              # for temporary use
            global expSignal
            expSignal=response[len(response)-4:len(response)]
            if expSignal[0] == "[" and expSignal[3] == "]":
                out += response[0:len(response)-4]
            else:
                out += response
        if inputType =="humidityInput":
            hum=input_content.split(";")
            humidity=hum[0]
            temperature=hum[1]
            if (float(temperature)>18):
                out+=f"警告警告,温度已达{temperature},烧死我啦，嘟嘟鲁。"
            if (float(humidity)>30):
                out+=f"警告警告,湿度已达{humidity},淹死我啦，嘟嘟鲁。"


            if (float(temperature)>28):
                out+=f"警告警告,温度已达{temperature},请调整至合适温度"
            global t1
            t2 = time.time()
            print(t2)
            print(t1)
            if float(humidity)>80:
                t1 = time.time()
            if t2-t1>10:
                timeInSecond = t2 - t1
                hours = int(timeInSecond / 3600)
                minutes = int((timeInSecond - hours * 3600) / 60)
                seconds = int(timeInSecond - hours * 3600 - minutes * 60)

                out+=f"警告警告,距离上次浇水时间已过去{hours}时{minutes}分{seconds}秒,请及时浇水"
    return out


max_length_record_Voice = 5


def keepReceiveMsg():
    while not exit:
        msg = getData()
        print(msg)
        processedMsg = handleMsg(msg)

        message=TTS(processedMsg)
        # mp3_to_wav(mp3_path)
        sendString(expSignal)
        time.sleep(0.01)
        sendWAV("temp.mp3")
        time.sleep(0.02)
        # sendMsg(processedMsg)


# def mainSendMsg():
#     


# build connection
s = socket.socket()
s.bind(("172.28.171.187", 9009))
# n+1
s.listen(5)
# block, build session, sock_clint
sock, addr = s.accept()

print(sock, addr)

tRec = threading.Thread(target=keepReceiveMsg(), name="Receive_Msg")
# tSend = threading.Thread(target=mainSendMsg(), name="MainSendMsg")

tRec.start()
# tSend.start()
tRec.join()
# tSend.join()
