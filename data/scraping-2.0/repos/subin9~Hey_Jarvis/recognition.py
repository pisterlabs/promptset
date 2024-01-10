import speech_recognition as sr
from TTS.api import TTS
import pvporcupine
import struct
import pyaudio
import wave
import random
from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts
from konlpy.tag import Okt
from openai import OpenAI
import requests
from datetime import datetime
import pyautogui
import cv2
import numpy as np
# import argparse
#
# args = argparse.ArgumentParser()
# args.add_argument("--weather_api", type=str, default=False)
# args.add_argument("--openai_api", type=str, default=False)
# args.add_argument("--porcupine_api", type=str, default=False)
# args = args.parse_args()

weather_authKey='gGryFchORUuq8hXITjVLWQ'
porcupine_api_key='4miWi+Z8pccHq/VEEHgq+n+ctXn4fTxFZBwaxnajrFCLT61WbVBRKg=='

# client = OpenAI(api_key=openai_api_key)

porcupine = pvporcupine.create(
    access_key=porcupine_api_key,
    keyword_paths=['자비스.ppn'],
    model_path='porcupine_params_ko.pv',
    sensitivities=[0.7]
)

pa = pyaudio.PyAudio()

audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

okt = Okt()

record_commands={'화면','녹화'}
record_length_s=set([str(i)+"초" for i in range(1,60)])
record_length_m=set([str(i)+"분" for i in range(1,60)])
record_length_h=set([str(i)+"시간" for i in range(1,24)])

station_dict={'서울':108, "원주":114, '동해':106, '대전':133, "안동":136, '전주':146, '대구':143, '광주':156, '부산':159, '여수':168}
station=set(station_dict.keys())
weather_commands={'날씨', '알려줘'}
weather_station=station

#TODO: 모델 다운로드를 마치면 API를 사용하지 않고 직접 훈련한 모델을 이용할 것
# config.load_json("/path/to/xtts/config.json")
# model = Xtts.init_from_config(config)
# model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True)

config = XttsConfig()
model=None


def stt():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source, timeout=4)

    return r.recognize_google(audio_data=audio, language='ko-KR')


def tts_api(text):
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    tts.tts_to_file(
        text=text,
        file_path="output.wav",
        speaker_wav="jarvis_source.mp3",
        language="en")
    speak("output.wav")


def speak(path):
    chunk = 1024
    f = wave.open(path, "rb")
    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                    channels=f.getnchannels(),
                    rate=f.getframerate(),
                    output=True)

    data = f.readframes(chunk)

    while data:
        stream.write(data)
        data = f.readframes(chunk)

    stream.stop_stream()
    stream.close()
    p.terminate()


def tts(text):
    outputs = model.synthesize(
        text,
        config,
        gpt_cond_len=3,
        language="en"
    )
    speak(outputs)


def gpt(command):
    # completion = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a Jarvis in Ironman movies, 2008"},
    #         {"role": "user", "content": "Answer like Jarvis " + command},
    #     ]
    # )
    # return completion.choices[0].message.content
    return


def check_command(text):
    result=set(okt.morphs(text))
    if result & record_commands:
        sec=result & record_length_s
        min=result & record_length_m
        hour=result & record_length_h
        if sec:
            sec=sec.pop()
            if len(sec)==2:
                sec=int(sec[0])
            else:
                sec=int(sec[0:2])
        else:
            sec=0
        if min:
            min=min.pop()
            if len(min)==2:
                min=int(min[0])
            else:
                min=int(min[0:2])
        else:
            min=0
        if hour:
            hour=hour.pop()
            if len(hour)==2:
                hour=int(hour[0])
            else:
                hour=int(hour[0:2])
        else:
            hour=0

        duration=sec+min*60+hour*3600
        return (1,duration)
    elif result & weather_commands:
        return (2,result & weather_station)
    else:
        return 0


def execute_command(command,api):
    command, args=command
    if command==1:
        if api:
            tts_api("Okay, start recording.")
            screen_size = pyautogui.size()

            codec = cv2.VideoWriter_fourcc(*"XVID")
            output = cv2.VideoWriter("screen_record.avi", codec, 10.0, screen_size)
            cnt=0
            while True:
                cnt+=1
                image = pyautogui.screenshot()
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output.write(frame)
                if cnt==args*10:
                    break

            # release the video writer and destroy the window
            output.release()
            cv2.destroyAllWindows()
            tts_api('Recording successfully done. File recorded as screen_record.avi')
        else:
            tts("녹화를 시작합니다.")
    elif command==2:
        if api:
            tts_api("Let me check the weather.")
        else:
            tts("Let me check the weather.")

        start = datetime.today().hour - 2
        end = datetime.today().hour - 1
        stn=args.pop()
        url = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php?tm1=" + str(start) + "&tm2=" + str(
            end) + "&stn=" + str(station_dict[stn]) + "&help=0&authKey=" + weather_authKey
        r = requests.get(url)
        text = r.text.split('\n')[-3].split()

        answer = "Current temperature is " + text[11] + " degrees. Humidity is " + text[13] + " percent."
        if text[15] == '0' and text[18] == '0':
            answer += " There is no rain or snow."
        elif text[15] == '0':
            answer += " There is a little rain. Take an umbrella."
        else:
            answer += " There is a little snow. Take an umbrella."
        if api:
            tts_api(answer)
        else:
            tts(answer)
    else:
        if api:
            tts_api(gpt(command))
        else:
            tts(gpt(command))


def wait(porcupine, pa, audio_stream, api=False, awaken=False):
    try:
        print("Listening...")
        while True:
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

            keyword_index = porcupine.process(pcm)

            if keyword_index >= 0:
                break

    except:
        print("Error")

    finally:
        print("Awaken")
        porcupine.delete()
        audio_stream.close()

        if awaken:
            if api:
                tts_api("Hi, I'm Jarvis. How can I help you?")
            else:
                tts("Hi, I'm Jarvis. How can I help you?")
        else:
            if api:
                tts_api(random.choice(["Yes, sir?",
                                       "What can I do for you?",
                                       "Welcome back. What's up?",
                                       "How can I help you?"]))
            else:
                tts(random.choice(["Yes, sir?",
                                   "What can I do for you?",
                                   "Welcome back. What's up?",
                                   "How can I help you?"]))

        command = check_command(stt())

        if command[0]>=0:
            execute_command(command,api)
        else:
            if api:
                tts_api(gpt(command))
            else:
                tts(gpt(command))

        wait(porcupine, pa, audio_stream, awaken=True)


if __name__ == '__main__':
    wait(porcupine, pa, audio_stream, api=True)