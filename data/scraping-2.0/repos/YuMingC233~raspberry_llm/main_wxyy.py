from __future__ import print_function

# iflytek
import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

# raspberry pi
from datetime import datetime
import ssl
import time
import wave
import alsaaudio
import websocket
from gpiozero import Button
from signal import pause
import signal
import sys
import threading
import os

# # # # #
#
# iflytek code
#
# # # # #

STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识


class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, AudioFile):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.AudioFile = AudioFile

        # 公共参数(common)
        self.CommonArgs = {"app_id": self.APPID}
        # 业务参数(business)，更多个性化参数可在官网查看
        self.BusinessArgs = {"domain": "iat", "language": "zh_cn", "accent": "mandarin", "vinfo": 1, "vad_eos": 10000}

    # 生成url
    def create_url(self):
        url = 'ws://ws-api.xfyun.cn/v2/iat'
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # 拼接鉴权参数，生成url
        url = url + '?' + urlencode(v)
        # print("date: ",date)
        # print("v: ",v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        # print('websocket url :', url)
        return url


# 收到websocket消息的处理
def on_message(ws, message):
    # 识别结果
    global over_result
    try:
        code = json.loads(message)["code"]
        sid = json.loads(message)["sid"]
        if code != 0:
            errMsg = json.loads(message)["message"]
            print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))

        else:
            data = json.loads(message)["data"]["result"]["ws"]
            # print(json.loads(message))
            result = ""
            for i in data:
                for w in i["cw"]:
                    result += w["w"]
            over_result += result
            # print("sid:%s call success!,data is:%s" % (sid, json.dumps(data, ensure_ascii=False)))

    except Exception as e:
        print("receive msg,but parse exception:", e)


# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)


# 收到websocket关闭的处理
def on_close(ws, a, b):
    print("### closed ###")


# 收到websocket连接建立的处理
def on_open(ws):
    def run(*args):
        frameSize = 8000  # 每一帧的音频大小
        intervel = 0.04  # 发送音频间隔(单位:s)
        status = STATUS_FIRST_FRAME  # 音频的状态信息，标识音频是第一帧，还是中间帧、最后一帧

        with open(wsParam.AudioFile, "rb") as fp:
            while True:
                buf = fp.read(frameSize)
                # 文件结束
                if not buf:
                    status = STATUS_LAST_FRAME
                # 第一帧处理
                # 发送第一帧音频，带business 参数
                # appid 必须带上，只需第一帧发送
                if status == STATUS_FIRST_FRAME:

                    d = {"common": wsParam.CommonArgs,
                         "business": wsParam.BusinessArgs,
                         "data": {"status": 0, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    d = json.dumps(d)
                    ws.send(d)
                    status = STATUS_CONTINUE_FRAME
                # 中间帧处理
                elif status == STATUS_CONTINUE_FRAME:
                    try:
                        d = {"data": {"status": 1, "format": "audio/L16;rate=16000",
                                      "audio": str(base64.b64encode(buf), 'utf-8'),
                                      "encoding": "raw"}}
                        ws.send(json.dumps(d))
                    except Exception as e:
                        print("发送数据时发生错误:", e)
                # 最后一帧处理
                elif status == STATUS_LAST_FRAME:
                    d = {"data": {"status": 2, "format": "audio/L16;rate=16000",
                                  "audio": str(base64.b64encode(buf), 'utf-8'),
                                  "encoding": "raw"}}
                    ws.send(json.dumps(d))
                    time.sleep(1)
                    break
                # 模拟音频采样间隔
                time.sleep(intervel)
        ws.close()

    thread.start_new_thread(run, ())


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

"""
录音方法
"""


def recoding():
    global recording_stopped
    global hash_filename

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
    print("请说话。")
    # The _period size_ controls the internal number of frames per period.
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


"""
结束录制方法
"""


def stop_recoding():
    global recording_stopped
    recording_stopped = True
    print("button pressed.\n")


"""
删除临时音频文件方法
"""


def delete_temp_file():
    global hash_filename
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
推送音频文件方法
"""


def push_media():
    # 语音识别参数
    global wsParam




    time1 = datetime.now()
    wsParam = Ws_Param(APPID='518780c2', APISecret='ZGIyNjY4OTY4MDA5ZjQxMWFkY2M5OTAx',
                       APIKey='b0782afd1d08c6093ffa6205a400010f',
                       AudioFile=hash_filename)
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl,on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open

    # error: Only http, socks4, socks5 proxy protocols are supported
    # 代理参数
    ws.run_forever(
        sslopt={"cert_reqs": ssl.CERT_NONE}
    )
    time2 = datetime.now()
    # print(time2 - time1)


"""
将结果推送至Open AI的GPT-4-turbo接口中
"""


def push_to_gpt4turbo():
    from openai import OpenAI
    global over_result

    client = OpenAI


if __name__ == "__main__":


    # 停止录制标志
    global recording_stopped
    # 录制时自动生成的基于时间的哈希文件名称
    global hash_filename
    # 语音识别结果
    global over_result

    # 初始化部分全局变量
    over_result = ""
    recording_stopped = False

    print("按下按钮结束说话，长按按钮结束程序。")

    # 当按钮被按下时，结束recoding方法
    button.when_pressed = stop_recoding
    # 当按钮长按时的方法引用
    button.when_held = handle_long_press

    recoding()
    print("录音结束，正在推送音频文件……")
    push_media()
    # 删除临时音频文件
    delete_temp_file()
    print("正在将结果推送至大语言模型……")

    pause()
