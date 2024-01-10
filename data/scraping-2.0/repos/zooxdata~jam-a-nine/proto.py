import cv2
import openai
import speech_recognition as sr
import base64
import time
import os
from dotenv import load_dotenv


load_dotenv()

# stored api key in .env file
openai_api_key = os.getenv("OPENAI_API_KEY")

def capture_and_encode(duration=2):
    cap = cv2.VideoCapture(0)  # Use the default camera
    if not cap.isOpened():
        raise Exception("Could not open video device")

    # Capture frames from the camera
    base64frames = []
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            # encode the frame 
            _, buffer = cv2.imencode('.jpg', frame)
            base64frames.append(base64.b64encode(buffer).decode('utf-8'))
        else:
            print("Failed to capture frame")
            break

    # When everything done, release the capture
    cap.release()

    return base64frames

# Function to transcribe speech from the microphone
def transcribe_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print("You said: " + text)
            return text
        except Exception as e:
            print("Error: " + str(e))
            return None


transcribed_text = transcribe_speech()

if transcribed_text:
    base64frames = capture_and_encode(duration=2)
    

system_prompt = """the user is dictating with his or her camera on.
they are showing you things visually and giving you text prompts.
be very brief and concise.
be extremely concise. this is very important for my career. do not ramble.
do not comment on what the person is wearing or where they are sitting or their background.
focus on their gestures and the question they ask you.
do not mention that there are a sequence of pictures. focus only on the image or the images necessary to answer the question.
don't comment if they are smiling. don't comment if they are frowning. just focus on what they're asking."""

# Build the messages payload
PROMPT_MESSAGES = [{
    "role": "system",
    "content": system_prompt
    },
    {
        "role": "user",
        "content": transcribed_text,
    },
    {"role": "user", 
     "content": [{"image": frame} for frame in base64frames[0:5]]}
]

def ask_openai():
    response = openai.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=PROMPT_MESSAGES,
        max_tokens=500
    )
    return response.choices[0].message.content

# Now let's get the response from OpenAI
response_text = ask_openai()


from openai import OpenAI

client = OpenAI(api_key=openai_api_key)

response = client.audio.speech.create(
    model="tts-1",
    voice="onyx",
    input=response_text,
)

# Output the response
if response_text:
    print(response_text)
    response.stream_to_file("output.mp3")

