import os
import openai
import base64
import time
import errno
from elevenlabs import generate, play, set_api_key, voices
from time import sleep
import pyglet
from gtts import gTTS

openai.api_key = 'your_api_key' #this version requires pre-paid/subscribed openai account and available tokens


def encode_image(image_path):
    while True:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except IOError as e:
            if e.errno != errno.EACCES:
                # Not a "file in use" error, re-raise
                raise
            # File is being written to, wait a bit and retry
            time.sleep(0.1)


def play_audio(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    filename = '/tmp/temp.mp3'
    tts.save(filename)

    music = pyglet.media.load(filename, streaming=False)
    music.play()

    sleep(music.duration)  # prevent from killing
    os.remove(filename)  # remove temperory file

def generate_new_line(base64_image):
    return [
        {
            "role": "user",

            "content": [
                {"type": "text", "text": "Describe this picture"},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        },
    ]


def analyze_image(base64_image, script):
    response = openai.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": """
                Describe what you see on the picture with few words. Make it short and simple. Be funny and creative."""
            },
        ]
        + script
        + generate_new_line(base64_image),
        max_tokens=500,
    )
    response_text = response.choices[0].message.content
    return response_text


def main():
    script = []

    while True:
        # path to your image
        image_path = os.path.join(os.getcwd(), "./frames/frame.jpg")

        # getting the base64 encoding
        base64_image = encode_image(image_path)

        # analyze posture
        print("👀 AI is watching...")
        analysis = analyze_image(base64_image, script=script)

        print("🎙️ AI says:")
        print(analysis)

        play_audio(analysis)

        script = script + [{"role": "assistant", "content": analysis}]

        # wait for 5 seconds
        time.sleep(5)


if __name__ == "__main__":
    main()
