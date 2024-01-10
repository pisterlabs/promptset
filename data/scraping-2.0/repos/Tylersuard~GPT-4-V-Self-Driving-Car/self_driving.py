import pyautogui
import random
import openai
import time
import base64
import requests
import os


time.sleep(10)

# OpenAI API Key
api_key = "YourOpenAIKey"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

#Prime the screenshots
for i in reversed(range(3)):
    pyautogui.screenshot(f'screenshot{i}.png',region=(282, 148, 680, 442)) #For the screen region, set where the window of your car simulator is.

while True:
    pyautogui.screenshot('screenshot0.png', region=(282, 148, 680, 442))
     # Take a screenshot
    base64_image0 = encode_image('screenshot0.png')

    base64_image1 = encode_image('screenshot1.png')

    base64_image2 = encode_image('screenshot2.png')

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "I am playing a game, and I need your help.  I am driving a car, and I need to know what to do next.  I have attached three screenshots of what I see.  The first screenshot is now, the second screenshot was taken one second ago, and the third screenshot was taken two seconds ago.  Please tell me what to do next.  Please press the W key to accelerate, the A key to turn left, the D key to turn right, or the S key to brake.  Return only a single character, W, A, D, or S, in square brackets [] followed by your reason for that decision.  The command will be applied for .5 seconds.  Please be conscious of the speed and direction of the vehicle.  I want to explore the city without crashing into anything.  Please do not go into the grass.  If you find yourself in the grass, please turn around and go back to the city."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image0}"
            }
          },
                    {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image1}"
            }
          },
                    {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image2}"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
}
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print(response.json())
        key = response.json()["choices"][0]["message"]["content"]
        key = key[key.index("[")+1:key.index("]")]
        print(key)
    except:
        time.sleep(5)
        continue

    if key == "W" or key == "S":
        pyautogui.keyDown(key) # Press the random key
        time.sleep(.25) # Wait for 1 second
        pyautogui.keyUp(key) # Release the key
        time.sleep(.75)

    elif key == "A" or key == "D":
        pyautogui.keyDown(key)
        pyautogui.keyDown("W")
        time.sleep(.25)
        pyautogui.keyUp("W")
        time.sleep(.75)
        pyautogui.keyUp(key[0])

    #delete screenshot2.png:
    os.remove('screenshot2.png')
    #rename screenshot1.png to screenshot2.png:
    os.rename('screenshot1.png', 'screenshot2.png')
    os.rename('screenshot0.png', 'screenshot1.png')
    time.sleep(4)
