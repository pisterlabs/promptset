import openai
import open_clip
import torch
from PIL import Image
import cv2
import json
from Captions import WatchVideo
from json_processing import process_video, write_output_to_file, read_captions_from_file
from VID2AUD import vid2aud
from openai_chat import generate_openai_chat_response
from audio_transcription import lemme_see

def main():
    # Define the video path
    video_path = "/Users/jimvincentwagner/tests/video_1677841652_k9Wj4Wkr2c.mp4"

    # Process the video and generate captions
    Captions = WatchVideo(video_path)

    # Load the captions from the JSON file
    txt = read_captions_from_file("output.json")

    # Combine the captions into a single string
    prompt_lines = "\n".join(txt)

    # MoviePy for audio extraction
    audio = vid2aud(video_path)

    # Get the audio transcription via zac script
    audio_info = lemme_see(lemme_see, top_n=3)  


    classes = audio_info

    # Check the audio transcription
    print(audio_info)

    # Generate the response from OpenAI Chat Completion
    response = generate_openai_chat_response(prompt_lines, str(audio_info))

    # Print the response
    print(response)

if __name__ == "__main__":
    main()
