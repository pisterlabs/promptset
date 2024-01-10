from __future__ import print_function

import os
import logging
from queue import Queue
import signal
from signal import pause
import threading
import time
import wave

from pydub import AudioSegment
from pydub.playback import play

import alsaaudio
from gpiozero import Button
from openai import OpenAI

# # # #
#
# Raspberry pi Code
#
# # # #

# 声卡上的按钮对应的BCM引脚号
button = Button(17)
# 默认声卡
device = 'default'
# 设置触发长按事件的时间阈值
button.hold_time = 1.0  # 长按时间设置为1秒
# 退出事件
exit_event = threading.Event()

msg = [
    {"role": "system", "content": "请使用更简洁的语句回答我的问题。"},
    {"role": "user", "content": "你好。今天的天气针不戳！😎😎😎"}
]

"""
录音方法
"""


def recoding():
    global recording_stopped

    # 使用哈希值作为文件名
    hash_filename = "./static/temp/" + hash(time.time()).__str__() + ".wav"

    f = wave.open(hash_filename, 'wb')

    # Open the device in nonblocking capture mode. The last argument could
    # 以非阻塞捕捉模式打开设备。最后一个参数也可以是
    # just as well have been zero for blocking mode. Then we could have
    # 零，代表阻塞模式。那样我们就可以不必要
    # left out the sleep call in the bottom of the loop
    # 在循环底部放置sleep调用
    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK, channels=1, rate=16000,
                        format=alsaaudio.PCM_FORMAT_S16_LE, periodsize=160, device=device)

    f.setnchannels(1)
    f.setsampwidth(2)  # PCM_FORMAT_S16_LE remains the same as it represents 16-bit sample width
    f.setframerate(16000)

    # print('%d channels, %d sampling rate\n' % (f.getnchannels(), f.getframerate()))
    # The period size controls the internal number of frames per period.
    # 周期大小控制每周期的内部帧数。
    # The significance of this parameter is documented in the ALSA api.
    # 这个参数的重要性在ALSA api文档中有说明。
    # For our purposes, it is suficcient to know that reads from the device
    # 对我们来说，只需知道从设备读取
    # will return this many frames. Each frame being 2 bytes long.
    # 将返回这么多帧。每个帧是2字节长。
    # This means that the reads below will return either 320 bytes of data
    # 这意味着下面的读取将返回320字节的数据
    # or 0 bytes of data. The latter is possible because we are in nonblocking
    # 或者0字节的数据。后者是可能的，因为我们处于非阻塞
    # mode.
    # 模式。

    while not recording_stopped:
        # Read data from device (设备读取数据)
        l, data = inp.read()

        if l:
            f.writeframes(data)
            time.sleep(.001)

    f.close()

    return hash_filename


"""
播放方法
"""


def play_result(file_path):
    song = AudioSegment.from_mp3(file_path)
    play(song)


"""
结束录制方法
"""


def stop_recoding():
    global recording_stopped
    recording_stopped = True
    # print("button pressed.\n")


"""
删除临时音频文件方法
"""


def delete_temp_file():
    import os
    import glob

    # 列出特定路径下的所有文件
    files = glob.glob('./static/temp/*')

    # 遍历列表中的每个文件路径，并删除它
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print(f"Error deleting file {f}: {e}")

    print("临时文件已删除")


"""
长按时退出程序方法
"""


def handle_long_press():
    delete_temp_file()
    print("程序退出。")
    # Send a SIGUSER1; this seems to cause signal.pause() to return.
    # 发送 SIGUSER1；这似乎能够让 pause() 返回。
    os.kill(os.getpid(), signal.SIGUSR1)


"""
stt api调用 (whisper)
"""


def stt(file_name):
    audio_file = open(file_name, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )
    return transcript


"""
对话方法
"""


def conversation(msg):
    result = ""

    # 生成对话
    stream = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=msg,
        stream=True,
    )

    print("他说：", end="")
    for chunk in stream:
        # 如果生成的文本不为空
        if chunk.choices[0].delta.content is not None:
            result += chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, end="")

    print()
    # 生成的文本转语音
    text_to_speech(result)

    msg.append({"role": "assistant", "content": result})

    return msg


def text_to_speech(text):
    # 连接到数据源
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=text
    )
    response.stream_to_file('./static/temp/temp_spoke.mp3')

    play_result('./static/temp/temp_spoke.mp3')


"""
主方法
"""
if __name__ == "__main__":
    # 设置pydub的日志级别为错误
    logging.getLogger("pydub.converter").setLevel(logging.INFO)
    # 设置代理环境变量
    os.environ['http_proxy'] = 'http://10.11.171.154:1234'
    os.environ['https_proxy'] = 'http://10.11.171.154:1234'

    # 停止录制标志
    global recording_stopped
    global client

    # 初始化必要变量
    client = OpenAI()
    msg = conversation(msg)

    while True:
        # 按按钮结束recoding 事件
        button.when_pressed = stop_recoding
        # 当按钮长按时 事件
        button.when_held = handle_long_press

        recording_stopped = False
        print("你说(按下按钮结束说话，长按按钮结束程序)：", end="")
        # 录制时自动生成的基于时间的哈希文件名称
        file_name = recoding()

        you_say = stt(file_name)
        print(you_say)

        # 删除临时音频文件
        delete_temp_file()
        msg.append({"role": "user", "content": you_say})
        msg = conversation(msg)

    pause()
