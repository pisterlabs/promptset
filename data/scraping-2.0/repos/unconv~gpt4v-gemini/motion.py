import modules.cv2_stream as cv2_stream
from playsound import playsound
from openai import OpenAI
import os

import modules.helpers as helpers

client = OpenAI()

stream_url = helpers.get_stream()

messages = [
    {
        "role": "system",
        "content": """You are an AI assistant that reports what it sees in short sentences. You are connected to a video feed and will be shown a collage of frames from a video and you will tell what what is happening in the video as if you are watching. Don't let the user know that you are seeing a sequence of images. Pretend as if you are a human seeing what is happening live. You are talking directly to the user. Assume any activity is performed by the user, who you shall refer to as "You". Example responses: "I see you drew an elephant" or "I see a dog" or "I see you put a TV remote on the table". Keep your responses very concise. If nothing particularly interesting happened since the previous image sequence, answer with the JSON {"status": "NO_CHANGE"}""".strip(),
    }
]

for collage in cv2_stream.detect_changes(stream_url):
    print("Motion detected!")
    collage.save("collage.jpg", format="JPEG")

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{helpers.image_b64(collage)}"
            }
        ]
    })

    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4-vision-preview",
        max_tokens=1024
    )

    response_message = response.choices[0].message

    if '{"status": "NO_CHANGE"}' in response_message.content:
        messages.pop()
        continue

    messages.append(response_message)

    audio = client.audio.speech.create(
        input=response_message.content,
        model="tts-1",
        voice="onyx",
    )

    audio.stream_to_file("audio.mp3")
    print("GPT: " + response_message.content)
    playsound("audio.mp3")
    os.remove("audio.mp3")

