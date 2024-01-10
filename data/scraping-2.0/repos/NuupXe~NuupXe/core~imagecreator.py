import configparser
import urllib.request
import json
import datetime
import os
import random
import time
import requests

from openai import OpenAI

class ImageCreator:

    def __init__(self, service='bing', market='en-US', resolution='1920x1080', output_directory='output'):
        self.service = service
        self.market = market
        self.resolution = resolution
        self.output_directory = output_directory
        self.wallpaper_name = 'image.jpg'

    def _download_bing_image(self):
        try:
            urllib.request.urlopen("http://google.com")
        except urllib.error.URLError as e:
            time.sleep(10)
        else:
            response = urllib.request.urlopen("http://www.bing.com/HPImageArchive.aspx?format=js&idx=0&n=1&mkt=" + self.market)
            obj = json.load(response)
            url = 'http://www.bing.com' + obj['images'][0]['urlbase'] + '_' + self.resolution + '.jpg'

            if not os.path.exists(self.output_directory):
                os.makedirs(self.output_directory)
            path = os.path.join(self.output_directory, self.wallpaper_name)

            if os.path.exists(path):
                today_date = datetime.datetime.now().strftime("%m/%d/%Y")
                file_date = time.strftime('%m/%d/%Y', time.gmtime(os.path.getmtime(path)))
                if today_date == file_date:
                    print("You already have today's Bing image")
                else:
                    print(f"Downloading Bing wallpaper to {path}")
                    with open(path, 'wb') as f:
                        bingpic = urllib.request.urlopen(url)
                        f.write(bingpic.read())
            else:
                print(f"Downloading Bing wallpaper to {path}")
                with open(path, 'wb') as f:
                    bingpic = urllib.request.urlopen(url)
                    f.write(bingpic.read())

    def _download_openai_image(self):
        ham_radio_prompts = [
            "Create an image of a ham radio operator communicating with Morse code.",
            "Generate an image of a vintage ham radio station with antennas.",
            "Show a ham radio enthusiast listening to signals on a shortwave radio.",
            "Picture an amateur radio operator making a contact during a contest.",
            "Create an image of a QSL card exchange between ham radio operators.",
            "Generate a scene of a ham radio shack with various transceivers and equipment.",
            "Illustrate a ham radio operator participating in a field day event.",
            "Show a ham radio operator using Morse code to transmit an emergency message.",
            "Create an image of a ham radio antenna tower reaching into the sky.",
            "Generate a scene of a ham radio club meeting with members discussing radios."
        ]

        random_prompt = random.choice(ham_radio_prompts)
        image_size = 1024

        services = configparser.ConfigParser()
        path = "configuration/services.config"
        services.read(path)
        api_key = services.get("openai", "api_key")
        client = OpenAI(api_key=api_key)
        response = client.images.generate(
            model="dall-e-3",
            prompt=random_prompt,
           size=f"{image_size}x{image_size}"
        )
        if 'url' in response.model_dump()['data'][0]:
            image_url = response.model_dump()['data'][0]['url']
            print(image_url)
            image_response = requests.get(image_url)
            if image_response.status_code == 200:
                local_file_path = "/tmp/image.jpg"
                with open(local_file_path, 'wb') as f:
                    f.write(image_response.content)
                print(f"Image saved to {local_file_path}")
            else:
                print("Failed to download the image from OpenAI.")
        else:
            print("No image URL found in the OpenAI response.")

    def create_image(self):
        if self.service == 'bing':
            self._download_bing_image()
        elif self.service == 'openai':
            self._download_openai_image()
        else:
            print("Invalid service selected. Choose 'bing' or 'openai'.")
