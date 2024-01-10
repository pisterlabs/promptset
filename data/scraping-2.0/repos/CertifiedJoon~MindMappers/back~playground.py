from time import sleep
from PIL import Image
import pytesseract
import openai
import json
from dotenv import load_dotenv
import os


load_dotenv()
openai.api_key = os.environ["OPEN_AI_KEY"]


def get_completion(
    prompt: str,
):  # Andrew mentioned that the prompt/ completion paradigm is preferable for this class
    model = "gpt-3.5-turbo"
    messages = [
        {
            "role": "system",
            "content": f"""
You are a bot that helps a librarian in daily tasks. The librarian is working to help visually impaired students.
Assist the librarian as best as possible.
""",
        },
        {"role": "user", "content": prompt},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


# Simple image to string
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
ocr_result = pytesseract.image_to_string(Image.open("./img/test.png"))

prompt = f"""
I have utilized OCR technology to read an old manuscript. But the content seems unclear due to
limitation of OCR technology. I am a librarian who has to turn this text into understandable natural language for visually impaired students.
I would like you to infer and guess the original content of the old manuscript and reconstruct
the given text into natural language. here is the text to interpret:

```
{ocr_result}
```

Return the output in the three paragraphs, keeping in mind that the output will be read directly to visually impaired students. Explain, in the begining of each paragraph, the purpose of the paragraph.
1. Brief introduction of the above text from OCR so that visually impaired student gets an idea before reading the text.
2. The text word by word. Edit only unnatural language into natural language.
3. Conclusion of the text so that visually impaired student know the content has ended.

"""

result = get_completion(prompt)
result = (
    "Hello, I am jane and I will be reading the text to you. ... Firstly, I will introduce you to the context. ... Secondly, I will read out the text. ... Thirdly, I will conclude the text. Here we go!\n "
    + result
)
print(result)


#####################################################

# import pyttsx3


# engine = pyttsx3.init()


# """ RATE"""
# rate = engine.getProperty("rate")  # getting details of current speaking rate
# print(rate)  # printing current voice rate
# engine.setProperty("rate", 150)  # setting up new voice rate


# """VOLUME"""
# volume = engine.getProperty(
#     "volume"
# )  # getting to know current volume level (min=0 and max=1)
# print(volume)  # printing current volume level
# engine.setProperty("volume", 1.0)  # setting up volume level  between 0 and 1

# """VOICE"""
# voices = engine.getProperty("voices")  # getting details of current voice
# # engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
# engine.setProperty(
#     "voice", voices[1].id
# )  # changing index, changes voices. 1 for female

# """Saving Voice to a file"""
# # On linux make sure that 'espeak' and 'ffmpeg' are installed
# engine.save_to_file("Hello World", "test.mp3")
# engine.runAndWait()

# engine.say(
# result
# )
# engine.runAndWait()

#####################################################
# import requests

# url = "https://api.genny.lovo.ai/api/v1/tts/sync"

# payload = {"speed": 1, "text": result, "speaker": "640f477d2babeb0024be422b"}
# headers = {
#     "accept": "application/json",
#     "content-type": "application/json",
#     "X-API-KEY": os.environ["LOVO_KEY"],
# }

# response = requests.post(url, json=payload, headers=headers)

# if response.status_code == 201:
#     res = json.loads(response.text)

#     if res["status"] == "done":
#         print(res["data"][0]["urls"][0])
#     else:
#         url = f"https://api.genny.lovo.ai/api/v1/tts/{res.id}"

#         headers = {
#             "accept": "application/json",
#             "X-API-KEY": os.environ["LOVO_KEY"],
#         }
#         i = 1
#         while res["status"] == "in_progress":
#             response = requests.get(url, headers=headers)
#             res = json.loads(response.text)
#             sleep(i)
#             i = int(i) * 1.5

#         print(res["data"][0]["urls"][0])
# else:
#     print(response.text)


import requests

CHUNK_SIZE = 1024
url = "https://api.elevenlabs.io/v1/text-to-speech/pNInz6obpgDQGcFmaJgB"

headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": os.environ["EL_KEY"],
}

data = {
    "text": result[:1700],
    "model_id": "eleven_monolingual_v1",
    "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
}

response = requests.post(url, json=data, headers=headers)
if response.status_code == 400:
    print(response.text)
else:
    print("success")
    with open("output.mp3", "wb") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
