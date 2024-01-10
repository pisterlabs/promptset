from IPython.display import display, Image, Audio

import cv2
import base64
import time
import openai
from openai import OpenAI
import os
import requests
import ast
import re

OPENAI_API_KEY = "your openai key"
client = OpenAI(api_key = OPENAI_API_KEY)
# OpenAI.api_key = os.getenv('OPENAI_API_KEY')
def extract_frames(video_path, interval=1):
    """
    Extract frames from the video at the specified interval.
    """
    video = cv2.VideoCapture(video_path)
    print("attempted extracted video")
    frames = []
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    while video.isOpened():
        frame_id = video.get(1)  # Current frame number
        success, frame = video.read()
        if not success:
            break
        if frame_id % (frame_rate * interval) == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()
    return frames

def analyze_frames_with_gpt4(frames, client):
    """
    Send frames to GPT-4 for analysis and return the descriptions.
    """

    # "Describe these video frames in terms of subject, action, and objects involved and format it like so (Subject: [subject], Action [action], Object [objects])."
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "Please describe these video frames in terms of subject, action, objects involved, and environment. Format each description as a dictionary in a Python list. Here is an example of the format I am looking for:\n\n"
                "[\n"
                "    {'subject': 'cat', 'action': 'sitting', 'objects': ['mat'], 'environment': 'living room'},\n"
                "    {'subject': 'dog', 'action': 'barking', 'objects': ['mailman'], 'environment': 'front yard'}\n"
                "]\n\n"
                "Now, please format the descriptions of the video frames in the same way:"
            ] + list(map(lambda x: {"image": x, "resize": 768}, frames))
        }
    ]

    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 500,
    }

    result = client.chat.completions.create(**params)
    return result.choices[0].message.content


def gpt_parse_events_batch(descriptions, batch_size):
    """
    Use GPT in a chat-style interaction to parse descriptions into structured data.
    Processes the descriptions in batches to adhere to API limits.
    """
    all_structured_events = []

    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i + batch_size]
        structured_events = gpt_parse_events(batch)
        all_structured_events.extend(structured_events)

    return all_structured_events


#def gpt_parse_events(descriptions):
    """
    Use GPT to parse a batch of descriptions into structured (subject, action, object(s)) tuples.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Please reformat each description into structured data with subject, action, and objects."}
    ]
    
    for desc in descriptions:
        messages.append({"role": "user", "content": desc})
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return process_gpt_responses(response['choices'][0]['message']['content'])

def gpt_parse_events(descriptions):
    """
    Use GPT to parse a batch of descriptions into structured (subject, action, object(s)) tuples.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Please reformat each description into structured data with subject, action, and objects."}
    ]
    
    for desc in descriptions:
        messages.append({"role": "user", "content": desc})
    
    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0  # Adjust if needed
    )

    # Extract the response content
    response_message = response.choices[0].message.content if response.choices else ""
    return process_gpt_responses(response_message)

# Rest of your code remains the same


def process_gpt_responses(response_text):
    """
    Process the GPT responses to extract (subject, action, object(s)) tuples.
    """
    events = []
    print("expected response", response_text)
    # Implement parsing logic based on the expected response format
    # Placeholder for parsing logic
    return events

def process_video_frames(video_frames):
    """
    Process the structured data from video frames into a list of events.
    """
    events = []

    for frame in video_frames:
        # Directly extract data from the dictionary
        subject = frame.get("subject", "")
        action = frame.get("action", "")
        objects = frame.get("objects", [])
        environment = frame.get("environment", "")

        event = {
            "subject": subject,
            "action": action,
            "objects": objects,
            "environment": environment
        }
        events.append(event)

    return events


#def parse_description_to_events(descriptions):
    """
    Parse the descriptions from GPT-4 to extract (subject, action, object(s)) tuples.
    """
    events = []
    for description in descriptions:
        # [Implementation of parsing the description to extract the required data]
        # ...
        pass
    return events

def append_timestamps(events, interval):
    """
    Append timestamps to each event.
    """
    timestamped_events = []
    for i, event in enumerate(events):
        timestamp = i * interval  # Assuming interval is in seconds
        timestamped_events.append((timestamp, event))
    return timestamped_events

def main():
    print("file exists?", os.path.exists('../../data/bison.mp4'))
    video_path = '../../data/bison.mp4'
    interval = 1  # Interval in seconds for frame extraction

    frames = extract_frames(video_path, interval)
    print("got frames")
    descriptions = analyze_frames_with_gpt4(frames, client)
    print("descriptions", descriptions)
    print("description type", type(descriptions))
    processed_events = process_video_frames(descriptions)
    print("processed_events", processed_events)
    # batch_size = 10 
    # events = gpt_parse_events_batch(descriptions, batch_size)
    # print("events", events)
    timestamped_events = append_timestamps(processed_events, interval)
    print("timestamped_events", timestamped_events)
    print("descriptions", descriptions)
    # [Save or process the timestamped events as needed]
    # ...
    # Assuming `client` is an initialized OpenAI client and `frames` is a list of base64-encoded frames


if __name__ == "__main__":
    main()
