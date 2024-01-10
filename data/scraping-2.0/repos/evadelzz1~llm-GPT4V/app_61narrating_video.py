from dotenv import load_dotenv
from openai import OpenAI
import cv2  # pip install opencv-python
import base64
import os

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)


# Initialize the OpenAI client with the API key
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

temp_video_name = "./data/football.mp4"
video = cv2.VideoCapture(temp_video_name)

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

base64_frames = []
frame_count = 0

# Read frames from the video in a loop
while True:
    # Capture frame-by-frame
    success, frame = video.read()
    
    # If the frame was not retrieved successfully, we have reached the end of the video
    if not success:
        break
    
    # Check if the frame is the one we want to capture (every 3rd frame)
    if frame_count % 3 == 0:
        
        # Convert the frame to JPEG format
        retval, buffer = cv2.imencode(".jpg", frame)
        
        # Convert the image buffer to base64
        frame_base64 = base64.b64encode(buffer)
        
        # Decode byte string into UTF-8 to get a string representation of base64
        frame_base64 = frame_base64.decode("utf-8")
        
        # Append the base64 string to the List
        base64_frames.append(frame_base64)
    
    # Increment the frame counter
    frame_count += 1

# When everything done, release the video capture object
video.release()
    
# Optional: Print the number of frames captured
print(f'Number of frames captured: {len(base64_frames)}')



PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "provided frames are from a video."
            "I would like you to provide a documentary like narration to this video."
            "Only provide the dramatic narration and nothing else."
            "not even anything in square brackets explaining the scene. just the narration.",
            *map(
                lambda x: {"image": x, "resize": 768},
                base64_frames[0::10],
            ),
        ],
    },
]

params = {
    "model": "gpt-4-vision-preview",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 1000,
    "stream": True,    
}

response = client.chat.completions.create(**params)

full_response = ""
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
        full_response += str(chunk.choices[0].delta.content)

speech_file_path = temp_video_name.replace(".mp4","") + "_narration.mp3"
response = client.audio.speech.create(
    model="tts-1",
    voice = "alloy",
    input = full_response,
)
response.stream_to_file(speech_file_path)