from flask import Flask, jsonify
from flask_cors import CORS

import threading
import pyaudio
import wave
import os
import openai
import json



# 函数：读取文件的文本内容
def read_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


# 使用 Python 的 Flask 框架来创建一个 Web 应用
app = Flask(__name__)
CORS(app) # 跨源资源共享（CORS）支持

# 录音相关的全局变量
is_recording = False        # 是否正在录音
frames = []                 # 用于存储录音的音频数据
audio = pyaudio.PyAudio()   # 创建 PyAudio 对象，用于录音
stream = None               # 用于存储 PyAudio 对象的流

FORMAT = pyaudio.paInt16    # 音频格式
CHANNELS = 1                # 声道数
RATE = 44100                # 采样率
CHUNK = 1024                # 采样点数
OUTPUT_FILENAME = "output.wav" # 输出文件名

def record_audio():
    global is_recording, frames, stream, audio
    # audio = pyaudio.PyAudio() # 创建 PyAudio 对象，用于录音
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK) # 打开流，传入响应参数

    while is_recording:
        data = stream.read(CHUNK)   # 读取采样点
        frames.append(data)         # 将采样点添加到 frames 中


@app.route('/start', methods=['GET'])
def start_recording():
    global is_recording, frames, stream
    if not is_recording:
        is_recording = True
        threading.Thread(target=record_audio).start()
        return jsonify(message="Recording started"), 200
    else:
        return jsonify(message="Recording is already in progress"), 400

@app.route('/stop', methods=['GET'])
def stop_recording():
    global is_recording, frames, stream
    if is_recording:
        is_recording = False

        stream.stop_stream()    # 停止流
        stream.close()          # 关闭流
        audio.terminate()       # 关闭 PyAudio 对象

        # 将 frames 中的音频数据写入WAV文件
        with wave.open(OUTPUT_FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)                       # 设置声道数
            wf.setsampwidth(audio.get_sample_size(FORMAT))  # 设置采样点数
            wf.setframerate(RATE)                           # 设置采样率
            wf.writeframes(b''.join(frames))                # 将 frames 写入文件

        print(f"File saved as {os.path.abspath(OUTPUT_FILENAME)}")

        frames = []  # 清空frames在这里，确保保存完音频后再清空
        stream = None
        
        # 使用 OpenAI API 进行语音识别
        api_key_content = read_from_file('api_key.txt')
        openai.api_key = api_key_content
        audio_file = open("output.wav", "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        print(transcript.text)

        # 读取JSON文件识别结果
        with open('config.json', 'r', encoding='utf-8') as file:
            config = json.load(file)

        # 读取后处理的JSON配置文件
        system_message = config["system_message"]
        user_message = config["user_message"]
        model_name = config["model_name"]
        max_taken_value = config["MaxTaken"]

        # 生成响应对话
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message+transcript.text},

        ]
        response = openai.ChatCompletion.create(model=model_name,
                                                messages=messages,
                                                max_tokens=max_taken_value,)

       # 输出结果
        token_count = response["usage"]["total_tokens"]                 # 花费的令牌数
        response_content = response['choices'][0]['message']['content'] # 响应内容
        print("Token count: " + str(token_count))
        print(response_content)

        return jsonify(message="Recording stopped", response=response_content), 200
    else:
        return jsonify(message="No recording in progress"), 400

if __name__ == '__main__':
    app.run(port=5000)
