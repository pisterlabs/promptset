import csv

# Path to your CSV file
csv_test = "csv_test.csv"
from base64 import b64decode
from pathlib import Path
import os
import io
from openai import OpenAI
import requests
import api_key

openaipi_api_key = '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'

client = OpenAI(api_key = openaipi_api_key)


def dalleImage(_number,input_prompt):
    # Set up our connection to the API.
    # select the engine using. (possible for stable diffusion 2)

    response = client.images.generate(
    model="dall-e-2",
    prompt="a realistic photograph based on the following describtion" + input_prompt + "the image should looks like a man made photo, \nDO NOT create CGI or 3D like images",
    size="1024x1024",
    quality="standard",
    n=1,
    )
    image_url = response.data[0].url


    # Send a GET request to the image URL
    response = requests.get(image_url)

    IMAGE_DIR = Path.cwd() / "dalle_image-0"
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Open a file in binary write mode

        fileName = str(_number) + "image.png"
        image_file = IMAGE_DIR / fileName
        with open(image_file, "wb") as file:
            # Write the content of the response to the file
            file.write(response.content)

    else:
        print(f"Failed to download the image. Status code: {response.status_code}")

        

# Open the CSV file for reading
with open(csv_test, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)

    # Skip the header row
    next(reader, None)

    # Iterate over each row in the CSV file
    for row in reader:
        cluster_number = row[0]  # First element in the row is the cluster number
        labels = row[1]  # second element in the row is the labels number
        input_prompt = row[2]    # Third element in the row is the input_prompt
        dalleImage(cluster_number,input_prompt)

