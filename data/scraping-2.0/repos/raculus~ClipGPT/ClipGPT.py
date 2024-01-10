import asyncio
from openai import AsyncOpenAI
import keyboard
import time
import clipboard
from pystray import MenuItem as item
import pystray
from PIL import Image
import threading
import os


def invert(image):
    return image.point(lambda p: 255 - p)


def invert_image(image):
    r, g, b, a = image.split()

    r, g, b = map(invert, (r, g, b))

    img2 = Image.merge(image.mode, (r, g, b, a))
    return img2


def exit():
    os._exit(0)


event = threading.Event()


def icon_rotate_threaded():
    image = invert_image(icon.icon)
    angle = 0
    while not event.is_set():
        icon.icon = image.rotate(angle)
        angle += 10
    icon.icon = invert_image(icon.icon)


f = open("API_KEY.txt", "r")
API_KEY = f.readline()
client = AsyncOpenAI(
    api_key=API_KEY,
)


async def question(msg: str):
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": msg,
            }
        ],
        model="gpt-3.5-turbo",
    )
    print(f"Answer: {chat_completion.choices[0].message.content}")
    return chat_completion.choices[0].message.content


def filter_code(answer: str):
    if "```" not in answer:
        return None
    splitted = answer.split("```")
    return (splitted[1], f"{splitted[0]}\n{splitted[2]}")


image = Image.open("gpt.png")
menu = (item("Exit", exit),)
icon = pystray.Icon("name", image, "ClipGPT", menu)


async def main():
    global event
    while True:
        if keyboard.is_pressed("ctrl+x") or keyboard.is_pressed("ctrl+c"):
            event = threading.Event()
            icon_rotate_thread = threading.Thread(target=icon_rotate_threaded)
            icon_rotate_thread.daemon = True
            icon_rotate_thread.start()

            time.sleep(0.01)
            clip_text = clipboard.paste()
            print(f"Question: {clip_text}")
            answer = await question(clip_text)
            filtered = filter_code(answer)
            if filtered is not None:
                code, answer = filtered
                clipboard.copy(answer)
                clipboard.copy(code)
            else:
                clipboard.copy(answer)

            event.set()


def main_threaded():
    asyncio.run(main())


main_thread = threading.Thread(target=main_threaded)
main_thread.daemon = True
main_thread.start()

icon.run()
