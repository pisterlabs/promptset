import os
import openai
from PIL import Image
import requests
import csv
import time
from io import BytesIO
from pathlib import Path

path = Path("/home/shivani/work/data/images")
openai.api_key = "key"

with open("/home/shivani/work/data/new_800.tsv", "r") as csvfile:
        writer = csv.reader(csvfile, delimiter="\t")
        for line in writer:
            sen = line[0]
            count = line[2]
            try:
                response = openai.Image.create(prompt=sen, n=1, size="512x512")

                image_url1 = response['data'][0]['url']

                response1 = requests.get(image_url1)

                img1 = Image.open(BytesIO(response1.content))
                image = "image"+ str(count) + ".png"
                img1.save(f"{path}/{image}")
            except openai.error.InvalidRequestError:
                 print("skipping this senetence")

            if int(count) % 40 == 0:
                time.sleep(300)


            


