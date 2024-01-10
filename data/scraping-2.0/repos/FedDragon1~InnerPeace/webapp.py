import base64
import io
import json
import os

import openai

import chain

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from flask import Flask
from flask import request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def emotion_detection():
    return np.random.rand(7)


@app.route('/api/chat', methods=['POST'])
@cross_origin()
def hello_world():
    data = json.loads(request.data)
    emotion = data["emotion"]
    information = data["data"]
    history = data["history"]

    ret = chain.get_chat_analysis(information, emotion, history)

    return ret


@app.route('/api/stt', methods=['POST'])
def get_stt():
    base64_audio = json.loads(request.data)["data"].split("base64,")[-1]
    audio = base64.b64decode(base64_audio)
    print(base64_audio)
    with open("temp123.ogg", "wb") as f:
        f.write(audio)
    print("ausydiauhsjhdkajhsd")
    transcript = openai.Audio.transcribe("whisper-1", open("temp123.ogg", "rb"), api_key=os.environ.get("OPENAI_API"))
    print(transcript)
    return transcript["text"]


@app.route('/api/emotion', methods=["POST"])
def get_emotion():
    base64_data = json.loads(request.data)["data"].split("base64,")[1]
    base64_decoded = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(base64_decoded))
    image_np = np.array(image)[:, 7:7+48, :-1]
    image_np = np.sum(image_np, axis=2)
    # plt.imshow(image_np, cmap="gray")
    # plt.show()
    image_np = image_np.reshape((1, 48, 48, 1))
    guess, ret = chain.emotion_detection(image_np)
    return ret


@app.route('/')
def main():
    return open("./frontend/index.html").read()


if __name__ == '__main__':
    app.run()
