import asyncio
import datetime
import os

from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
import json

import config
from openai_wrapper import OpenAIWrapper
from speech_recognizer import SpeechRecognizer
from voice_synthesizer import VoiceSynthesizer

config = config.Config()
recognizer = SpeechRecognizer(config, config.recognizer)
synthesizer = VoiceSynthesizer(config, config.voice_synthesizer)
ai = OpenAIWrapper(config.openai_api_key)


class DataHandler:
    send_audio = False

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.send_audio = False

    # テキストデータの処理
    async def text_proc(self, text: str):
        # テキストデータを音声データに変換
        try:
            if text == "":
                return

            ret = await ai.query(text, config.ai_key)
            result = ret[0]
            token = ret[1]
            lapse_time = ret[2]
            if text != "":
                await self.send_message(result + " (" + str(token) + " tokens) " + str(lapse_time) + "秒")
                ai.add_history("user", text)
                ai.add_history("system", result)

            filepath = ""
            if self.send_audio:
                # 音声データを作成
                filepath = await self.text_to_audio(result)
                # 音声データを送信
                with open(filepath, "rb") as file:
                    await self.send_audio(file.read())
                # 音声データを削除
                os.remove(filepath)

        except Exception as e:
            print(e)
            await self.send_text(e)
            return
        print("filepath = " + filepath)

    async def text_to_audio(self, text: str):
        # 現在時刻からファイル名を生成
        filename = datetime.datetime.now().strftime("synth%Y%m%d%H%M%S") + ".wav"
        filepath = "audio/" + filename

        start_time = datetime.datetime.now()
        synthesizer.synthesize(text, filepath)
        lapse_time = datetime.datetime.now() - start_time
        lapse_time = round(lapse_time.total_seconds(), 2)

        # 処理時間を表示
        time_text = "wav作成:" + synthesizer.config.voice_synthesizer + " (" + str(lapse_time) + "秒)"
        await self.send_text(time_text)
        return filepath

    async def handle_text(self, text: str):
        print("Received text data" + text)
        # テキストデータの処理
        response = "Processed: " + text
        await self.send_text(response)

    async def handle_message(self, text: str):
        print("Received text data" + text)
        await self.text_proc(text)
        return text

    async def handle_json(self, data: dict):
        print("Received JSON data" + str(data))
        # JSONデータの処理
        response = json.dumps({"processed": data})

        # jsonから "speaker" : true があれば、音声データを作成
        if "speaker" in data:
            if data["speaker"]:
                self.send_audio = True
            else:
                self.send_audio = False
        await self.send_text(response)

    async def handle_image(self, image_data: bytes):
        print("Received image data")
        # 画像データの処理（ここでは例示のため単純な応答を返す）
        response = "Image data received"
        await self.send_text(response)

    async def handle_audio(self, audio_data: bytes):
        print("Received audio data")

        # この関数の実行時間を計測
        received_time = datetime.datetime.now()
        result = recognizer.recognize_audio_data(audio_data)
        lapse_time = datetime.datetime.now() - received_time
        if result is None:
            return ""

        text = result[0]
        if text == "":
            return ""

        # 小数点以下2桁まで表示
        lapse_time = round(lapse_time.total_seconds(), 2)
        # 認識結果を表示
        time_text = text + " (" + str(lapse_time) + "秒)"
        await self.send_text(time_text)
        # 0.1秒の遅延で非同期処理
        await asyncio.sleep(0.1)
        await self.text_proc(text)
        return text

    async def send_text(self, text: str):
        print("Sending text data" + text)
        await self.websocket.send_text("text")
        await self.websocket.send_text(text)

    async def send_message(self, text: str):
        await self.websocket.send_text("message")
        await self.websocket.send_text(text)

    async def send_json(self, data: dict):
        await self.websocket.send_text("json")
        await self.websocket.send_text(json.dumps(data))

    async def send_image(self, image_data: bytes):
        await self.websocket.send_text("image")
        await self.websocket.send_bytes(image_data)

    async def send_audio(self, audio_data: bytes):
        print("sending audio data")
        await self.websocket.send_text("audio")
        await self.websocket.send_bytes(audio_data)
        print("sending audio data end")
