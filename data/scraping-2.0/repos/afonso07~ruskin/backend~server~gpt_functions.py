import os
from typing import Any, Dict, List, Union

from openai import OpenAI
from .gpt_contexts import critique_context
import base64
from elevenlabs import generate, play  # type: ignore

client = OpenAI()


def generate_image_prompt(imageURI: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this painting."},
                {
                    "type": "image_url",
                    "image_url": imageURI,
                },
            ],
        },
    ]


def generate_analysis_prompt(imageURI: str) -> str | None:
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": critique_context,
            },
        ]
        + generate_image_prompt(imageURI),
        max_tokens=500,
    )
    response_text = response.choices[0].message.content

    return response_text


if __name__ == "__main__":
    with open("test_images/manu_painting2.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    analysis_prompt = generate_analysis_prompt(encoded_string.decode())
    # print(encoded_string)

    audio = generate(
        text=analysis_prompt, voice=os.getenv("VOICE_ID"), model="eleven_turbo_v2"
    )
    play(audio)
