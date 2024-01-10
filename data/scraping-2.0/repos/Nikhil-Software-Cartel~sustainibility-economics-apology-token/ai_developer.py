import os
from openai import OpenAI
from dotenv import load_dotenv
import base64
import mimetypes
from PIL import ImageGrab
from typing import Literal
import io

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def image_to_base64(source) -> str:
    # Guess the MIME type of the image
    # print(source.__class__.__name__)

    if type(source) == str:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image'):
            raise ValueError("The file type is not recognized as an image")

        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        img64 = f"data:{mime_type};base64,{encoded_string}"

    if source.__class__.__name__ == 'DibImageFile':
        # Read the image binary data
        # convert to png
        buffer = io.BytesIO()
        source.save(buffer, format="PNG")
        img = buffer.getvalue()
        # Encode the binary data
        img64 = base64.b64encode(img).decode('utf-8')
        img64 = f"data:image/png;base64,{img64}"
        # print(img64)

    return img64


def image_to_html(
    img64, prompt="please describe this image in detail, give all info.", detail: Literal["low", "high"] = "high"
):
    # convert to png

    img64 = image_to_base64(img64)
    print("LOG:STATE: Generation in progress...")

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {"type": "image_url", "image_url": {"url": img64, "detail": detail}},
                ],
            }
        ],
        max_tokens=4096,
    )
    resp = response.choices[0].message.content
    print(resp)
    with open("output.html", "w") as f:
        f.write(resp)
    return resp


if im := ImageGrab.grabclipboard():
    prompt = """
    CONTEXT: Client wants to build a website like the image, you are senior developer
    UI: Make it ultra world class and modern and use best typography and colors
    FRAMEWORK: we are using sveltkit
    IMPORTANT: give only code, no need explanation or greetings!!!
    LANGUAGE: html only
    INSTRUCTIONS:
    BE smart create repated items using Javascript/Jquery looping
    1. i have given you an image, using this as reference make full website.
    2. i need html, css, js(optional) make all improvements possible, use placeholder images like https://placehold.co/600x400 , 
    3. wherever text needs to be written intelligently rewrite it, use dummy icons font awesome.
    4. we have appropriate license of all copyrighted material.
    5. make full clone of this, do not be afraid of lengthy code. maximum resemblence. else no use of your work. business critical!!!
    URLLIBS: https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/js/bootstrap.bundle.min.js https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/css/bootstrap.min.css https://pro.fontawesome.com/releases/v5.15.0/css/all.css
    VIDEO: https://www.dropbox.com/s/wy317udwi0k293i/video.mp4?raw=1"""
    # prompt = "Whats there in the image?"
    image_to_html(
        im,
        prompt=prompt,
    )
else:
    print("No image in clipboard")


# im.save('clipboard.png', 'PNG')
# check if image is of type
# print(im.__class__.__name__)
# print(image_to_base64(im))

# base64_string = image_to_base64("c:/Users/User/Downloads/mobilesdealrelos-reference.jpg")
