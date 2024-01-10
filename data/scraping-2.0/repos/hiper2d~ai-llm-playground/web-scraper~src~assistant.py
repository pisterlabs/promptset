import os
import subprocess

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import base64

load_dotenv(find_dotenv())
model = OpenAI()


def image_b64(image):
    with open(image, "rb") as f:
        return base64.b64encode(f.read()).decode()


def read_from_page(url):
    if os.path.exists("screenshot.jpg"):
        os.remove("screenshot.jpg")

    result = subprocess.run(
        ["node", "screenshot.js", url],
        capture_output=True,
        text=True
    )

    exitcode = result.returncode
    output = result.stdout

    if not os.path.exists("screenshot.jpg"):
        print("ERROR: Cannot read URL or create a screenshot.")

    b64_image = image_b64("screenshot.jpg")

    print("User questions: Making a screenshot of the page.")
    response = model.chat.completions.create(
        model='gpt-4-vision-preview',
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{b64_image}"
                    },
                    {
                        "type": "text",
                        "text": "Extract all movies from the screenshot that are in horror (ужасы) "
                                "or fiction (фантастика) genre. "
                                "Return a list of all horror and fiction movies on the screenshot with their genres and directors."
                    }
                ]
            }
        ],
        max_tokens=2048,
    )

    message = response.choices[0].message
    message_text = message.content
    print(message_text)

    print("User questions: What horror movies are there?")
    response = model.chat.completions.create(
        model='gpt-3.5-turbo-1106',
        messages=[
            {
                "role": "assistant",
                "content": message_text
            },
            {
                "role": "user",
                "content": "What horror movies are there?"
            }
        ],
        max_tokens=2048,
    )

    message = response.choices[0].message
    message_text = message.content
    print(message_text)

    return message_text


if __name__ == '__main__':
    read_from_page("https://rutracker.org/forum/viewforum.php?f=252")
