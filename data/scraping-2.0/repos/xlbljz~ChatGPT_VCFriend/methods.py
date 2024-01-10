import datetime
import json
import httpx
import time
import subprocess
from loguru import logger

import xml.etree.cElementTree as ET

import openai
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech.audio import AudioOutputConfig

from WXBizMsgCrypt3 import WXBizMsgCrypt
from config import *

speech_config = speechsdk.SpeechConfig(
    subscription=KEY, region=REGION)
wxcpt = WXBizMsgCrypt(WECOM_TOKEN, WECOM_AESKEY, WECOM_COMID)


def user_voice2_text(input_file_path):
    # 调用 voice_convert 函数将 WAV 文件转换为 AMR 格式
    voice_convert(input_file_path + '.amr', input_file_path + '.wav', 'wav')

    audio_input = speechsdk.AudioConfig(filename=input_file_path + '.wav')
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_input)

    done = False

    def stop_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognizing.connect(
        lambda evt: print('RECOGNIZING: {}'.format(evt)))
    speech_recognizer.recognized.connect(
        lambda evt: print('RECOGNIZED: {}'.format(evt)))
    speech_recognizer.session_started.connect(
        lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(
        lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(
        lambda evt: print('CANCELED {}'.format(evt)))
    # stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)

    speech_recognizer.stop_continuous_recognition()

    # 输出识别结果
    for result in speech_recognizer:
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Recognized speech: {}".format(result.text))
            return result.text

        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized.")
            # return 'No speech could be recognized.'
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech recognition canceled: {}".format(
                cancellation_details.reason))
            # return 'Speech recognition canceled: {}".format(cancellation_details.reason)'


def communicate_with_chatgpt(text):
    openai.api_key = OPENAI_KEY
    logger.debug(f'chatgpt received {text}')
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": text}
                ],
                # prompt=text,
                temperature=0.7,
                max_tokens=150,
                top_p=1,
                frequency_penalty=1,
                presence_penalty=0.1,
            )
            output_words = eval(
                f"u\'{response['choices'][0]['message']['content']}\'")
            logger.debug(f'chatgpt response {output_words}')
            break
        except openai.error.RateLimitError:
            time.sleep(0.1)

    return output_words


def chatgpt_response2_voice(text):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file_path = f"voice_cache/output/{now}"

    speech_config.speech_synthesis_language = "zh-CN"
    # speech_config.speech_synthesis_voice_name = "zh-CN-XiaoyouNeural"
    audio_config_output = speechsdk.audio.AudioOutputConfig(
        use_default_speaker=False)
    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config_output)
    result = speech_synthesizer.speak_text_async(text).get()
    stream = speechsdk.AudioDataStream(result)
    stream.save_to_wav_file(output_file_path + '.wav')

    # 调用 voice_convert 函数将 WAV 文件转换为 AMR 格式
    voice_convert(output_file_path + '.wav', output_file_path + '.amr', 'amr')

    return output_file_path


def voice_convert(input_path, output_path, fmt):
    # 使用 FFmpeg 工具将 WAV 文件转换为 AMR 格式
    try:

        cmd = ['ffmpeg', '-i', input_path, '-ar', '8000',
               '-ab', '12.2k', '-ac', '1', '-f', fmt, '-']

        with open(output_path, 'wb') as output_file:
            subprocess.run(cmd, stdout=output_file,
                           stderr=subprocess.PIPE, check=True)

    except subprocess.CalledProcessError as e:
        # 打印错误输出
        print(e.stderr.decode())
        exit(1)


def verify_url(request):
    sverify_msgsig = request.args.get('msg_signature')
    sverify_timestamp = request.args.get('timestamp')
    sverify_nonce = request.args.get('nonce')
    sverify_echostr = request.args.get('echostr')

    ret, sechostr = wxcpt.VerifyURL(
        sverify_msgsig, sverify_timestamp, sverify_nonce, sverify_echostr)

    if (ret != 0):
        print("ERR: VerifyURL ret: " + str(ret))

    else:
        print("done VerifyURL")

    return sechostr


def xml_parse(request):
    try:
        # 微信服务器发来的三个get参数
        timestamp = request.args.get("timestamp")
        nonce = request.args.get("nonce")
        encrypted_bytes = request.data
        print(request.data)
        if encrypted_bytes:
            # 获取msg_signature参数
            msg_signature = request.args.get("msg_signature")
            # 用微信官方提供的SDK解密，附带一个错误码和生成明文
            ierror, decrypted_bytes = wxcpt.DecryptMsg(
                encrypted_bytes, msg_signature, timestamp, nonce)
            # 若错误码为0则表示解密成功

            if ierror == 0:
                # 对XML进行解析
                xml_tree = ET.fromstring(decrypted_bytes)
                xml_dict = {
                    elem.tag: elem.text for elem in xml_tree.iter()}
                print(xml_dict)
                return xml_dict
            else:
                print('xml解析错误')
        else:
            print('encrypted_bytes为空')

    except Exception as e:
        print(e)


def msg_download(media_id):
    # 使用企业微信 API 接收消息
    # 获取token
    r = httpx.get(
        f'https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={WECOM_COMID}&corpsecret={APP_SECRET}').text
    js = json.loads(r)
    global access_token
    access_token = js['access_token']

    try:
        # 下载消息内容
        params = {
            'access_token': access_token,
            'media_id': media_id
        }

        # 下载数据直到下载完毕
        with httpx.stream("GET", "https://qyapi.weixin.qq.com/cgi-bin/media/get", params=params) as response:
            if response.status_code == 200:
                # 保存语音文件
                now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                input_file_path = f"voice_cache/intput/{now}"
                with open(input_file_path, "wb") as f:
                    # 将下载的数据写入文件
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                print("下载成功，路径为：", input_file_path)
                return input_file_path

            else:
                print("下载失败，错误码为：", response.json()[
                      "errcode"], response.json()["errmsg"])

    except Exception as e:
        print(e)


def find_key(json_obj, query_key):
    """
    递归查找 JSON 中的某个字段并返回该字段的值
    """
    if isinstance(json_obj, dict):
        # 遍历字典
        for key, value in json_obj.items():
            if key == query_key:
                return value
            else:
                result = find_key(value, query_key)
                if result is not None:
                    return result

    elif isinstance(json_obj, list):
        # 遍历列表
        for item in json_obj:
            result = find_key(item, query_key)
            if result is not None:
                return result

    return None
