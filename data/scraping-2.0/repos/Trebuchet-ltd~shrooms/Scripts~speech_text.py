import json
from base64 import b64decode
from pathlib import Path

import speech_recognition as sr
import openai

from DPT.run_monodepth import run

openai.api_key = "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"

DATA_DIR = Path.cwd() / "responses"

IMAGE_DIR = Path.cwd() / "images"
OUT_DIR = Path.cwd() / "output"

DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

def find_background(speech):
    ask = f"what is the background location of the scene: '{speech}'"

    response = openai.Completion.create(engine="text-davinci-001", prompt=ask, temperature=0.9,
                                        max_tokens=150,
                                        top_p=1,
                                        frequency_penalty=0.0,
                                        presence_penalty=0.6,
                                        )

    text = response['choices'][0]['text']

    return text


def find_points(speech):
    ask = f"find the different actions happening in this scene as different sentences : '{speech}'"

    response = openai.Completion.create(engine="text-davinci-001", prompt=ask, temperature=0.9,
                                        max_tokens=150,
                                        top_p=1,
                                        frequency_penalty=0.0,
                                        presence_penalty=0.6,
                                        )

    text = response['choices'][0]['text'].split('.')

    return text



def create_scene(background_prompt, action_prompts):

    response = openai.Image.create(
        prompt=background_prompt,
        n=1,
        size="512x512",
        response_format="b64_json",
    )
    for index, image_dict in enumerate(response["data"]):
        image_data = b64decode(image_dict["b64_json"])
        image_file = IMAGE_DIR / f"background_{index}.png"
        with open(image_file, mode="wb") as png:
            png.write(image_data)

    cnt = 0
    for prompt in action_prompts:
        if prompt:
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="512x512",
                response_format="b64_json",
            )
            for index, image_dict in enumerate(response["data"]):
                image_data = b64decode(image_dict["b64_json"])
                image_file = IMAGE_DIR / f"{cnt}.png"
                cnt += 1
                with open(image_file, mode="wb") as png:
                    png.write(image_data)


def main():
    # speech = "I had a dream , I was on mars, I had a lightsaber in my hand. suddenly i was fighting darth vader."
    # back_text = find_background(speech)
    # action_prompts = find_points(speech)
    # print(back_text)
    # print(action_prompts)


    # create_scene(back_text, action_prompts)
    model_weights = '../DPT/weights/dpt_hybrid-midas-501f0c75.pt'
    model_type = 'dpt_hybrid'
    run(str(IMAGE_DIR), str(OUT_DIR), model_weights, model_type,True )

# r = sr.Recognizer()
# speech = 'i saw a whale'
# with sr.Microphone() as source:
#     r.adjust_for_ambient_noise(source)
#
#     print("Please say something")
#
#     audio = r.listen(source)
#     recognition =r.recognize_google(audio)
#     print(recognition)
#     speech = recognition
#     # print("you have said: ", recognition)


if __name__ == "__main__":
    main()
