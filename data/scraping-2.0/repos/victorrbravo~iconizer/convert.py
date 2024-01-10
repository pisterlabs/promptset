#!/usr/bin/env python3
#  Author: Victor Bravo victor.bravo@wizeline.com
#  Description: This script will convert an image to text and then to an icon
#  Usage: python convert.py input.jpg output.jpg
#  Dependencies: pip install openai, pip install pillow, pip install requests
#  Notes: This script is using the OpenAI API, you need to have an account and an API key
#  https://beta.openai.com/
#  https://beta.openai.com/docs/api-reference
#  https://beta.openai.com/docs/introduction
#  https://beta.openai.com/docs/developer-quickstart/your-first-request
#  https://beta.openai.com/docs/developer-quickstart/authentication
#  https://beta.openai.com/docs/developer-quickstart/environment-variables
#  https://beta.openai.com/docs/developer-quickstart/python



import base64
from io import BytesIO
from PIL import Image
import sys
from openai import OpenAI
import os
import requests
import tempfile
import secrets
import string
import argparse

def decode_base64_image(base64_string):
    # Remove data prefix if present
    base64_string = base64_string.replace("data:image/jpeg;base64,", "")

    # Decode base64 into bytes
    image_bytes = base64.b64decode(base64_string)

    # Create an image from the bytes
    image = Image.open(BytesIO(image_bytes))

    return image

def encode_image_to_base64(image_path):
    try:
        # Open the image file
        with open(image_path, "rb") as image_file:
            # Read the image data
            image_data = image_file.read()

            # Encode the image data in Base64
            encoded_data = base64.b64encode(image_data)

            # Convert bytes to a UTF-8 string
            base64_string = encoded_data.decode("utf-8")

            return base64_string

    except Exception as e:
        print(f"Error: {e}")
        return None

def image_to_text(image_path, client):
    image_64 = encode_image_to_base64(image_path)
    image_url = f"data:image/jpeg;base64,{image_64}"
    response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[{"role": "user", "content": [{ "type": "text", "text": "Describe the image.Complete the final paragraph if this is necessary"},
                                           {"type": "image_url", "image_url": { "url": image_url} } ]
               }],
    max_tokens=100,
    )
    value_text = response.choices[0].message.content
    return value_text

def text_to_icon(value_text, client):
    image_size = 1024
    prompt_text = f"{value_text}.\n Your role is an assistant, identify the most important element of previous sentences and draw it as an ICON, you have to use only vectorial elements like lines and circles."
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt_text,
        size=f"{image_size}x{image_size}"
    )
    model_data = response.model_dump()
    print("model_data", model_data)
    image_url = model_data['data'][0]['url']
    return image_url

def download_image(url, local_filename):
    try:
        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        # Open the local file for writing in binary mode
        with open(local_filename, 'wb') as file:
            # Write the content to the local file in chunks
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Image downloaded successfully and saved at {local_filename}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")

def process_image(image_base64, output_file_path):
    try:
            if not image_base64:
                print("Error: image_base64 is empty.")
                return

            # Decode base64 to image
            img = decode_base64_image(image_base64)

            # Save the image as JPEG
            img.save(output_file_path, 'JPEG')

            print(f"JPEG image saved successfully at {output_file_path}")

    except Exception as e:
        print(f"Error: {e}")

def resize_image(input_path, output_path):
    try:
        # Open the image file
        image = Image.open(input_path)

        # Get the original size
        original_size = image.size

        # Calculate the new size (50% reduction)
        new_size = (int(original_size[0] * 0.25), int(original_size[1] * 0.25))

        # Resize the image
        resized_image = image.resize(new_size)

        # Save the resized image
        resized_image.save(output_path)

        print(f"Image resized successfully and saved at {output_path}")

    except Exception as e:
        print(f"Error resizing image: {e}")


def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(secrets.choice(characters) for _ in range(length))
    return random_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform a photo into an icon using OpenAI. It also generates a description for the input image')

    parser.add_argument('-f', '--filename', help='Enter the file path for the JPEG image.', required=True)
    parser.add_argument('-o', '--output', help='Specify the file path for the generated icon.', required=True)
    args = vars(parser.parse_args())
    print("args", args)
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        # or you can explicitly pass in the key (NOT RECOMMENDED)
        api_key=os.getenv("OPENAI_KEY"),
        )

    input_image = args['filename']
    output_image = args['output']

    temp_directory = tempfile.gettempdir()

    temp_image = f"{temp_directory}/temp_{ generate_random_string(8)}"
    value_text = image_to_text(input_image,client)
    print("DESCRIPTION:\n", value_text, '\n--------------------\n')
    image_url = text_to_icon(value_text, client)
    print("tem_image", temp_image)
    download_image(image_url, temp_image)
    resize_image(temp_image, output_image)
    print(f"Image resized to {output_image}")




