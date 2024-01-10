import cv2
from ultralytics import YOLO
import numpy as np
import torch
import os
import openai
from elevenlabs import clone, generate, play, set_api_key, voices
from elevenlabs.api import History
import requests

def request(prompt):
    endpoint = 'https://api.together.xyz/inference'
    res = requests.post(endpoint, json={
        "model": "togethercomputer/llama-2-70b-chat",
        "max_tokens": 1024,
        "prompt": prompt,
        "request_type": "language-model-inference",
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": [
            "[/INST]",
            "</s>"
        ],
        "sessionKey": "2e59071178ae2b05e68015136fb8045df30c3680"
    }, headers={
        "Authorization": "Bearer a38cd76484726df05bfc70c5b951f1f11c59dd5e0a1b191d2f407f6fd4326838",
    })
    return res.json()['output']['choices'][0]['text'].strip()

set_api_key("8f96a58113b07003fcf761c98bfb2c3b")
voice = voices()[1]
model = YOLO("yolov8m.pt").to(torch.device("mps"))
cap = cv2.VideoCapture(0)
openai.api_key = "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"

class_map = {}
with open("yolo_classes.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        class_map[i] = line.strip()

frame_num = 0
with torch.inference_mode():
    while True:
        frame_num += 1
        ret, frame = cap.read()
        results = model(frame)
        result = results[0]
        bboxes = result.boxes.xyxy
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        sizes = []
        for cls, bbox in zip(classes, bboxes):
            (x, y, x2, y2) = bbox
            sizes.append(abs(x2 - x) * abs(y2 - y))
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0, 2))
            cv2.putText(frame, class_map[cls], (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        if len(classes) > 0:
            pairs = [(i, cls, size) for i, (cls, size) in enumerate(zip(classes, sizes))]
            _, best_class, _ = max(pairs, key=lambda p: p[-1])


        if not ret:
            break
        print(frame_num % (30 * 5))
        interval = 1.5
        if frame_num % (30 * interval) >= 30 * interval - 10:
            cv2.putText(frame, "calculating!!!", (70, 150), cv2.FONT_HERSHEY_PLAIN, 10, (0, 255, 0), 5)
        cv2.imshow("Img", frame)
        if frame_num % (30 * interval) == 0:
            prompt = f"In two sentence, please describe the following object, and then tell me a short fun fact it. Start with 'This is a <object name>'. OBJECT={class_map[best_class]}"
            message = request(prompt)
            # completion = openai.ChatCompletion.create(
            #     model="gpt-3.5-turbo",
            #     messages=[
            #         {"role": "system", "content": "You are a helpful assistant."},
            #         {"role": "user", "content": f"In two sentence, please describe the following object, and then tell me a short fun fact it. Start with 'This is a <object name>'. OBJECT={class_map[best_class]}"}
            #     ]
            # )
            # message = completion.choices[0].message['content']
            print(message)
            # audio = generate(text=message, voice=voice)
            # play(audio)

        key = cv2.waitKey(1)
        if key == 27:
            break
cap.release()
cv2.destroyAllWindows()

