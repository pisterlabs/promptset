import cv2 
import easyocr
import pyautogui
import numpy as np

import openai
import pyperclip
import time
from PIL import Image

api_key = input("Enter API Key")
openai.api_key = api_key


def get_completion(prompt, model="gpt-3.5-turbo"):

    messages = [{"role": "system", "content": prompt}]

    response = openai.ChatCompletion.create(

    model=model,

    messages=messages,

    temperature=0,

    )

    return response.choices[0].message["content"]


reader = easyocr.Reader(['en'])


while True:

    ss = pyautogui.screenshot()
    ss.save("ss.png")

    img = cv2.imread("ss.png")
    img = img[1700:1850 , 1150:1800]
    cv2.imwrite("img.png" , img)

    result = reader.readtext('img.png')

    msg = ""
    notmsg = ""

    for detection in result: 
        text = detection[1] 
        for alpha in text:

            if alpha.isalpha() == True:
                msg += alpha
            elif alpha == " ":
                msg += " "
            else:
                notmsg += alpha

    msg = msg.replace("am" , "\n")
    msg = msg.replace("pm" , "\n")

    print(f"Message : {msg}")
    if msg != "":
        prompt = f"You are a human teenager, Reply infomally without emojis, dont give definite reply to '{msg}'"
        response = get_completion(prompt)
        print(f"Reposne : {response}\n\n")


        pyautogui.moveTo(700, 950, duration = 1)
        pyautogui.click()
        pyautogui.write(response)
        pyautogui.press("enter")
    time.sleep(10)





