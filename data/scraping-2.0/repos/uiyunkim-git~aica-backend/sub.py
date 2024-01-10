import requests
from queue import Queue
from time import sleep
import openai
import threading
import pyaudio
import tempfile
import wave
import json

DEEPL_API_KEY = '54b2312e-e3e9-1334-418e-bbce189c4b90:fx'
openai.api_key = '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'

gpt_buffer = ""
stt_buffer = ""
audio_data_buffer = []

FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1               # Number of audio channels (1 for mono, 2 for stereo)
RATE = 44100               # Sample rate (samples per second)
CHUNK = 1024                # Number of frames per buffer
SIZE_STT_BUFFER = 5

p = pyaudio.PyAudio()

def record():
    stop_recording = threading.Event()

    def audio_callback(in_data, frame_count, time_info, status):
        global audio_data_buffer
        in_data_copy = in_data[:]
        audio_data_buffer.extend(in_data_copy)

        return (in_data, pyaudio.paContinue)

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    stream_callback=audio_callback)

    stream.start_stream()

    def stop_recording_thread():
        input("Press Enter to stop recording...\n")
        stop_recording.set()
        record_thread.join()
        stream.stop_stream()
        stream.close()
        p.terminate()

    record_thread = threading.Thread(target=stop_recording_thread)
    record_thread.start()

def speech_to_text():
    global stt_buffer
    def stt_thread():
        global stt_buffer
        while True:
            stt_buffer = get_text(audio_data_buffer)
            print(f"STT Buffer: {stt_buffer}")
    stt_thread = threading.Thread(target=stt_thread)
    stt_thread.start()

def gpt_knowledge_bot():

    def gpt_thread():
        global gpt_buffer
        while True:
            try:
                gpt_buffer = gpt_powered_str_prefix_join()

            except Exception as e:
                print(e)
                gpt_buffer = "failed"
            print(f"GPT Concat: {gpt_buffer}")

    gpt_thread = threading.Thread(target=gpt_thread)
    gpt_thread.start()

def translate():
    global stt_buffer
    def translate_thread():
        global stt_buffer
        while True:
            translated_text = translate_text(stt_buffer, "EN")
            print(f"Translated: {translated_text}")

    translate_thread = threading.Thread(target=translate_thread)
    translate_thread.start()

def transcribe():
    text = None
    while True:
        new_text = get_text(audio_data_buffer)
        translated_text = translate_text(new_text, "EN")
        print(text, translated_text)

def get_text(audio_data_buffer):
    try:

        with wave.open("temp.wav", 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(bytearray(audio_data_buffer))
            print(f'frames: {len(audio_data_buffer)}')
        with open("temp.wav","rb") as f:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file= f,
                response_format="text"
            )
        text = response
        return text
    except Exception as e:
        print(e)
        return e

def translate_text(text, target_language):
    deepl_url = 'https://api-free.deepl.com/v2/translate'
    params = {
        'text': text,
        'target_lang': target_language,
        'auth_key': DEEPL_API_KEY,
    }
    response = requests.post(deepl_url, data=params)

    translation_data = response.json()
    translations = translation_data.get('translations', [])
    if translations:
        return translations[0].get('text', '')

def gpt_powered_str_prefix_join():
    sentences = '\n'.join([str(i + 1) + '. ' + item for i, item in enumerate(stt_buffer)])
    prompt = f"다음 주어진 {SIZE_STT_BUFFER}개의 문장은 사람이 말한 내용이야. 내용을 요약하지 않고 {SIZE_STT_BUFFER}개의 문장을 취합해줘.: {sentences} \n\n 결과물:"

    response = openai.completions.create(
        model="text-davinci-003",
        prompt=f"{prompt}",
        # messages=[
        #     {"role": "system", "content": "You are a helpful assistant. Your response should be in JSON format."},
        #     {"role": "user", "content": f"{prompt}"}
        # ],
        # response_format={"type": "json_object"}
        max_tokens=400,
    )

    return str(response.choices[0].text)

    # return json.loads(response.choices[0].message.content)["result"]

def circular_append_to_list(l, item):
    l.append(item)
    if len(l) > SIZE_STT_BUFFER:
        l.pop(0)
    return l

if __name__ == "__main__":
    record()
    sleep(1)
    speech_to_text()
    # gpt_knowledge_bot()
    translate()
