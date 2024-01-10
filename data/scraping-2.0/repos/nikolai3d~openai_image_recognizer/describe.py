import base64
import os
import asyncio
import aiohttp
import sys
import pygame
import cv2
import uuid

from pathlib import Path

async def camera_loop():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam, is there a webcam connected?")
    
    saved_path = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the resulting frame
        cv2.imshow('Camera Preview', frame)

        # Press 'q' to exit the loop
        pressed_key = cv2.waitKey(1) & 0xFF
        print (pressed_key)
        if pressed_key == ord('q'):
            break
        
        if pressed_key == ord('c'):
            saved_path = f'photo-{uuid.uuid4()}.png'
            cv2.imwrite(saved_path, frame)
            captured = True
            break
        
        # await asyncio.sleep(0.1)  # Allow other tasks to run 
        

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return saved_path

async def play_mp3_async(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)  # Allow other tasks to run 

# OpenAI API Key, get from OPENAI_API_KEY environment variable:

# Get OpenAI API Key from environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key is None:
    raise RuntimeError("OPENAI_API_KEY is not set in the environment variables.")
else:
    print("OpenAI API Key is successfully retrieved.")


# Function to encode the image

def encode_image(i_image_path):
    """
    Encodes an image file as a base64 string.

    Args:
      image_path (str): The path to the image file.

    Returns:
      str: The base64-encoded string representation of the image.
    """
    with open(i_image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def fetch(session, url, headers, payload):
    async with session.post(url, headers=headers, json=payload) as response:
        return await response.json()

async def fetch_raw_response(session, url, headers, payload):
    async with session.post(url, headers=headers, json=payload) as response:
        return await response.read()

async def describe_image(aio_session, local_image_path):
    # Getting the base64 string
    base64_image = encode_image(str(local_image_path))

    url = "https://api.openai.com/v1/chat/completions"

    print ("Sending request to OpenAI API...")
    print (f"Image path: {local_image_path}")
    print (f"API key: {openai_api_key}")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}",
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Describe this image in about two sentences"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 500,
    }

    fetch_task = asyncio.create_task(fetch(aio_session, url, headers, payload))

    while not fetch_task.done():
        print('.', end='', flush=True)
        await asyncio.sleep(1)  # Adjust the sleep time as needed

    response_object = await fetch_task


    print('Request completed')
    # print(response_object)

    # if there's an error in the response, print it and exit
    if "error" in response_object:
        print(response_object["error"]["message"])
        sys.exit(-1)

    # response_object = response.json()
    choices = response_object["choices"]
    if len(choices) > 0:
        text = choices[0]["message"]["content"]
        return text
    else:
        print("Unexpected response from OpenAI API")
        print(response_object)
        sys.exit(-1)

async def generate_narration(aio_session, input_text):
    url = "https://api.openai.com/v1/audio/speech"

    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {openai_api_key}",
    }
    payload = {
      "model": "tts-1",
      "input": f"{input_text}",
      "voice": "nova"
    }

    fetch_task = asyncio.create_task(fetch_raw_response(aio_session, url, headers, payload))

    while not fetch_task.done():
        print('*', end='', flush=True)
        await asyncio.sleep(1)  # Adjust the sleep time as needed

    response_bytes = await fetch_task

    saved_narration_path = f'narration-{uuid.uuid4()}.mp3'

    with open(saved_narration_path, 'wb') as file:
        file.write(response_bytes)

    print('\nRequest completed')

    return saved_narration_path
    # response_object.stream_to_file("output.mp3")

async def play_file(file_path):
    
    print(f'\nPlaying file {file_path}')
    play_task = asyncio.create_task(play_mp3_async(file_path))

    while not play_task.done():
        print('>', end='', flush=True)
        await asyncio.sleep(1)  # Adjust the sleep time as needed

    print('\nPlaying file done')


async def main():
    image_path = await camera_loop()
    print (f"Acquired image: {image_path}")

    if not image_path:
        print("No image captured, exiting...")
        return 0
    

    async with aiohttp.ClientSession() as session:
        description = await describe_image(session, image_path)

        print("--------------------")
        print(description)
        print("--------------------")

        narration_file = await generate_narration(session, description)

        await play_file(narration_file)

        return 0
    
# Running the main coroutine
asyncio.run(main())
