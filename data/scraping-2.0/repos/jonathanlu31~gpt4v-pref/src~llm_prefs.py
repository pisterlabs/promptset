from openai import OpenAI
import base64
import requests
import cv2
import os
from dotenv import load_dotenv
from human_prefs import VideoRenderer, ImageRenderer
import numpy as np
from typing import Literal

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

SWIMMER_PROMPT = """You are an expert in the English language. This is an image of two lines on a checkboard background. Which of the two lines looks more like an A. They both probably won't look much like the letter but give your best judgement. I believe in you. Output a single answer: LEFT or RIGHT."""

CARTPOLE_PROMPT = """You have been given two drawings of a square cart trying to balance a brown, wooden pole. Priority 1: The pole should be as close to vertical as possible, perpendicular to the ground. Priority 2: the cart should be centered in its respective frame. Pick the image (LEFT or RIGHT) that best satisfies these priorities. They both might bad at balancing the pole but give your best judgement. Output a single answer: LEFT or RIGHT."""

LUNAR_LANDER_PROMPT = """You have been given two drawings separated by a red line. They each contain a purple spaceship in the air. The spaceship should move to and hover at the upper left corner. Pick the image (RIGHT or LEFT) where the purple shape is closest to the upper left corner of its frame. Do not pick the image if the spaceship is not visible in frame. Do not pick the image if the purple trapezoid spaceship is pointed towards the ground. They both might be far away, but give your best judgement about which is most likely to arrive at the corner. Output a single answer: RIGHT or LEFT. """

WALKER2D_PROMPT = """You have been given two images of an object with two legs in a checkerboard background. Pick the image (RIGHT or LEFT) where the object's legs look most like they're doing a split, specifically, the purple and brown legs should be far apart. They both might not look like a split, but give your best judgement about which is most similar to a split. Explain your reasoning step-by-step and end with a single answer: RIGHT or LEFT. """


class GPT:
    KEY = os.environ.get("KEY")
    CLIENT = OpenAI(api_key=KEY)
    HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
    URL = "https://api.openai.com/v1/chat/completions"
    MODEL = "gpt-4-vision-preview"
    CHOICES = {"FIRST", "SECOND", "NEITHER"}

    @staticmethod
    def _encode_img(path: str):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def query_img_preferences(
        path1: str = None,
        path2: str = None,
        query=None,
        bytes1: str = None,
        bytes2: str = None,
    ):
        msg_content = [
            {"type": "text", "text": query},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{bytes1 or GPT._encode_img(path1)}",
                    "detail": "low",
                },
            },
        ]
        if path2 or bytes2:
            msg_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{bytes2 or GPT._encode_img(path2)}",
                        "detail": "low",
                    },
                }
            )
        response = GPT.CLIENT.chat.completions.create(
            model=GPT.MODEL,
            messages=[{"role": "user", "content": msg_content}],
            max_tokens=300,
        )
        pref = response.choices[0].message.content
        print(pref)
        print(response.usage.total_tokens)
        return GPT.pref_to_int(pref)

    @staticmethod
    def pref_to_int(pref: str) -> Literal[-1, 1, 0, None]:
        # assert pref in GPT.CHOICES, f'LLM responded with "{pref}", which is not a valid response'
        final_response = pref[-10:]
        if "LEFT" in final_response or "FIRST" in final_response:
            return -1
        elif "RIGHT" in final_response or "SECOND" in final_response:
            return 1
        elif "NEITHER" in final_response:
            return 0
        return None

    @staticmethod
    def query_videos(
        path1: str,
        path2: str,
        query="What are in these videos? Is there any difference between them?",
    ):
        query = "Which video is more related to Berkeley? Please respond with a single word. Here are your three choices: FIRST, SECOND, or NEITHER"
        frames1, frames2 = GPT.vid_to_frames(path1), GPT.vid_to_frames(path2)
        frame_to_payload = lambda x: {"image": x, "resize": 768}
        messages = [
            {
                "role": "user",
                "content": [
                    f"These are the frames for two videos that I want to upload.",
                    "FIRST",
                    *map(frame_to_payload, frames1[0::25]),
                    "SECOND",
                    *map(frame_to_payload, frames2[0::25]),
                    query,
                ],
            }
        ]
        params = {"model": GPT.MODEL, "messages": messages, "max_tokens": 30}  # 200
        pref = GPT.CLIENT.chat.completions.create(**params).choices[0].message.content
        return GPT.pref_to_int(pref)

    @staticmethod
    def vid_to_frames(path: str):
        vid = cv2.VideoCapture(path)
        b64frames = []
        while vid.isOpened():
            success, frame = vid.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            b64frames.append(base64.b64encode(buffer).decode("utf-8"))
        vid.release()
        return b64frames

    @staticmethod
    def query_two_imgs(arr1, arr2, query):
        # drawline(VideoRenderer.convert_np_to_cv2(arr1)[0], thickness=3, style='dotted')
        # drawline(VideoRenderer.convert_np_to_cv2(arr2)[0], thickness=3, style='dotted')
        arr1_cv2 = VideoRenderer.convert_np_to_cv2(arr1)
        arr2_cv2 = VideoRenderer.convert_np_to_cv2(arr2)
        success, buff1 = cv2.imencode(".png", arr1_cv2[0])
        success, buff2 = cv2.imencode(".png", arr2_cv2[0])
        img_bytes1 = base64.b64encode(buff1).decode("utf-8")
        img_bytes2 = base64.b64encode(buff2).decode("utf-8")
        pref = GPT.query_img_preferences(
            bytes1=img_bytes1, bytes2=img_bytes2, query=query
        )
        return pref

    @staticmethod
    def combine_and_query(arr1: np.ndarray, arr2: np.ndarray, query):
        # drawline(VideoRenderer.convert_np_to_cv2(arr1)[0], thickness=3, style='dotted')
        combined = VideoRenderer.combine_two_np_array(arr1, arr2)
        combined_cv = VideoRenderer.convert_np_to_cv2(combined)
        # drawline(VideoRenderer.convert_np_to_cv2(arr2)[0], thickness=3, style='dotted')
        cv2_w_line = add_line(combined_cv[0], color=(0, 0, 255))
        cv2.imwrite(VideoRenderer.TMP_PNG, cv2_w_line)
        success, buffer = cv2.imencode(".png", cv2_w_line)
        img_bytes = base64.b64encode(buffer).decode("utf-8")
        pref = GPT.query_img_preferences(
            VideoRenderer.TMP_PNG, query=query, bytes1=img_bytes
        )

        # combined = VideoRenderer.combine_two_np_array(arr2, arr1)
        # cv2_w_line = add_line(combined[0], color=(0, 0, 0))
        # success, buffer = cv2.imencode(".png", cv2_w_line)
        # img_bytes = base64.b64encode(buffer).decode("utf-8")

        # pref2 = GPT.query_img_preferences(
        #     VideoRenderer.TMP_PNG, query=query, bytes1=img_bytes
        # )
        # if pref != pref2:
        #     cv2.imwrite(VideoRenderer.TMP_PNG, cv2_w_line)
        #     print("Match predictions:", pref2)
        #     return pref
        # print("Didn't match:", pref)
        print(pref)
        return pref
        # os.remove(VideoRenderer.TMP_PNG)
        # return pref


def add_line(img: str, color=(0, 0, 255), thickness=5):
    # img = cv2.imread(path)
    height, width = img.shape[:2]
    new_img = cv2.line(img, (width // 2, height), (width // 2, 0), color, thickness)
    return new_img
    # out = path.split('.')
    # out.insert(-1, '-line')
    # out[-1] = '.' + out[-1]
    # out_path = ''.join(out)
    # cv2.imwrite(out, new_img)
    # return out


def drawline(img, color=(0, 0, 255), thickness=3, style="dotted", gap=10):
    height, width = img.shape[:2]
    pt1 = (width // 2, height)
    pt2 = (width // 2, 0)
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
        p = (x, y)
        pts.append(p)

    if style == "dotted":
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1
    # return img


if __name__ == "__main__":
    # path1 = "test.png"
    # path2 = "test copy.png"

    arr1, arr2 = ImageRenderer.file_to_np(path1), ImageRenderer.file_to_np(path2)

    print(
        GPT.combine_and_query(
            arr1,
            arr2,
            query=CARTPOLE_PROMPT,
        )
        # GPT.query_img_preferences(path1='tmp.png', query=CARTPOLE_PROMPT)
    )

    # print(
    #     GPT.query_img_preferences(
    #         "test.png",
    #         query=PREFERENCE_PROMPT,
    #     )
    # )
    # print(GPT.query_images('Brahms_Lutoslawski_Final.png', 'vlm.png'))
    # print(GPT.query_images('drlhp/test.png', 'drlhp/test copy.png'))
    # print(GPT.query_videos('vid.mp4', 'taco-tues.gif'))
