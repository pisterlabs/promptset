from dotenv import load_dotenv
from bs4 import BeautifulSoup as bs
import requests
from urllib.request import urlopen
import os
from openai import OpenAI
from moviepy.editor import VideoFileClip

load_dotenv() 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

video_info = {}
dish_path = 'test'
video_path = f'{dish_path}/video.mp4'
caption_path = f"{dish_path}/caption.txt"
output_path = f"{dish_path}/output.md"

def get_completion(prompt, model="gpt-3.5-turbo"):
    """
    Description: Access GPT
    """
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=model,
    messages=messages,
    max_tokens=500,
    temperature=0)
    return response.choices[0].message.content

def clean_url(url):
    """
    Description: Cleans URL by removing suffix
    """
    if "?" in url:
        return url[:url.index("?")]
    else:
        return url

def get_caption(url):
    """
    Description: Scrapes caption using bs4, cleans using GPT, and stores in caption.txt
    """
    response = requests.get(url)
    soup = bs(response.content, 'html.parser')
    caption = soup.find('meta', attrs={'name': 'description'})

    prompt=f"""Reformat the following caption, adding line breaks where necessary.

    Caption: ```{caption}```

    """

    output = get_completion(prompt)

    video_info['caption'] = output
    print("(1) caption successfully extracted.")

def download_video(url):
    """
    Description: Download video to local storage
    """
    headers = {
        'authority': 'ssstik.io',
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'dnt': '1',
        'hx-current-url': 'https://ssstik.io/en',
        'hx-request': 'true',
        'hx-target': 'target',
        'hx-trigger': '_gcaptcha_pt',
        'origin': 'https://ssstik.io',
        'referer': 'https://ssstik.io/en',
        'sec-ch-ua': '"Chromium";v="113", "Not-A.Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36',
    }

    params = {
        'url': 'dl',
    }

    data = {
        'id': url,
        'locale': 'en',
        'tt': 'YVdqMUE0',
    }

    response = requests.post('https://ssstik.io/abc', params=params, headers=headers, data=data)
    downloadSoup = bs(response.text, "html.parser")
    downloadLink = downloadSoup.a["href"]
    mp4 = urlopen(downloadLink)
    print("(2) video successfully downloaded.")
    with open(video_path, "wb") as output:
        while True:
            data = mp4.read(4096)
            if data:
                output.write(data)
            else:
                break

def convert_mp4_to_mp3(video_path, audio_file):
    """
    Description: Get audio from mp4
    """
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_file)

    video.close()
    audio.close()

    print(f"(3) {video_path} successfully converted to mp3.")

def transcribe(audio_path):
    """
    Description: Transcribe audio file
    """
    audio_file = open(audio_path, "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text")
    print(f"(4) {audio_path} successfully transcribed.")
    video_info['transcript'] = transcript

def prompt_ingredients():
    prompt=f"""Based on the following transcript and caption, output the dish being cooked. Then, identify whether there is a list of ingredients \
    provided in the transcript or caption. If the answer is yes, output the ingredients list word for word. If the answer is no, create a list of all \
    of the ingredients mentioned and how much of each ingredient is used. If there is no mention of how much of an ingredient is used, don't \
    include one. Do not make up a numerical quantity if it is not explicitly stated. Make sublists if necessary.

    Transcript: ```{video_info['transcript']}```
    Caption: ```{video_info["caption"]}```

    """

    output = get_completion(prompt)
    with open(output_path, "w") as file:
        file.write(output)
        print("(6) dish successfully written.")
        print("(7) ingredients successfully written.")


def prompt_procedure():
    prompt=f"""Using the following transcript and caption, write a series of enumerated steps, similar \
    to the procedure of a recipe. Be concise but retain important details. Each step should include (a) \
    the instruction, (b) the name and quantity of ingredients involved, (c) any relevant notes, and  \
    (d) indicators that the user should move on to the next step. Don't include any of these components \
    multiple times to minimize redundancy.

    Do not use outside information. Only use the information in the transcript and caption. Be as specific \
    as possible. 

    Example step: Add beef to the pan. Flatten the pieces and leave it alone. Once the beef has cooked around \
    the edges, remove it from the pan. It will finish cooking with the sauce and you don't want it to be overcooked. 


    Transcript: ```{video_info['transcript']}```
    Caption: ```{video_info['caption']}```

    """ 

    output = get_completion(prompt)

    with open(output_path, "a") as file:
        file.write("""

Procedure:  
""")
        file.write(output)
        print("(8) procedure successfully written.")


def main():
    raw_url = "https://www.tiktok.com/@cassyeungmoney/video/7212456384850332970"
    url = clean_url(raw_url)
    get_caption(url)
    download_video(url)

    convert_mp4_to_mp3(video_path, f'{dish_path}/audio.mp3')
    transcribe(f'{dish_path}/audio.mp3')
    print(video_info)
    prompt_ingredients()
    prompt_procedure()

if __name__ == '__main__':
    main()