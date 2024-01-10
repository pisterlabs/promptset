import base64
import requests
import cv2
import json
from pathlib import Path
import openai
import os
# OpenAI API Key
api_key = "sk-****"

# Function to convert a frame to base64
def encode_frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

# Function to read video and convert to base64
def get_base64_frames(video_path):
    video = cv2.VideoCapture(video_path)
    base64_frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        base64_frames.append(encode_frame_to_base64(frame))
    video.release()
    return base64_frames

# Function to make an API call to OpenAI
def analyze_frame(base64_image, headers):
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "These are frames from a video that I want to upload. Generate a compelling description that I can upload along with the video."
            }, {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }]
        }],
        "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()



def main():
    video_path = "data/testshort.mp4"
    frames = get_base64_frames(video_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    all_contents = []
    for i, base64_image in enumerate(frames):
        if i % 30 == 0:  # Analyze every 30th frame
            response_json = analyze_frame(base64_image, headers)
            content = response_json['choices'][0]['message']['content']
            all_contents.append(content)
    
    # Ensure the output directory exists
    output_dir = 'output_audio'
    os.makedirs(output_dir, exist_ok=True)

    # Write all contents to a file
    with open(os.path.join(output_dir, 'analyzed_contents.txt'), 'w', encoding='utf-8') as file:
        for content in all_contents:
            file.write(content + "\n\n")

    print(all_contents)

    # Now handle text-to-speech for all contents
    max_length = 2048
    assert all(isinstance(x, str) for x in all_contents), "All contents must be strings."
    all_contents_str = ' '.join(all_contents)
    parts = [all_contents_str[i:i + max_length] for i in range(0, len(all_contents_str), max_length)]

    for i, part in enumerate(parts, start=1):
        payload = {
            "model": "tts-1",
            "input": part,
            "voice": "onyx",
        }

        response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers=headers,
            json=payload,
        )

        if response.status_code == 200:
            part_file = os.path.join(output_dir, f"output_part_{i}.mp3")
            with open(part_file, "wb") as out:
                out.write(response.content)
            print(f"Audio content written to {part_file}")
        else:
            print(f"Request failed with status code {response.status_code}: {response.text}")


if __name__ == "__main__":
    main()
