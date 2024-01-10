import os
import shutil
from screenshotone import Client, TakeOptions
import typer
import uuid
import base64

from openai import OpenAI

app = typer.Typer()

ACCESS_KEY = os.environ.get('SCREENSHOTONE_ACCESS_KEY')
SECRET_KEY = os.environ.get('SCREENSHOTONE_SECRET_KEY')

@app.command()
def screenshot(url: str, filename: str = None):
    client = Client(ACCESS_KEY, SECRET_KEY)

    options = (TakeOptions.url(url)
        .format("jpg")
        .viewport_width(1024)
        .full_page(True)
        .block_cookie_banners(True)
        .block_chats(True))

    image = client.take(options)
    
    random_filename = filename if filename else uuid.uuid4()
    
    pathname = f"./data/{random_filename}.jpg"

    with open(pathname, 'wb') as result_file:
        shutil.copyfileobj(image, result_file)

        print(f"Saved {pathname}")
        return pathname

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@app.command()
def feedback(url: str):
    screenshot_file = screenshot(url)
    base64_image = encode_image(screenshot_file)

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are an expert in web design, ux and copyrighting. Give critical feedback on the website in screenshot in the image_url as a bulleted list."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    app()