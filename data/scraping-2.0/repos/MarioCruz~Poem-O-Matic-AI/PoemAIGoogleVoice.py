import requests
import cv2
import os
import openai
import platform
import time
from google.cloud import texttospeech

# Azure setup
API_KEY = os.getenv('AZURE_API_KEY')
ENDPOINT = 'https://mario-image-to-text.cognitiveservices.azure.com/'
DESCRIBE_URL = f'{ENDPOINT}/vision/v3.1/describe'
ANALYZE_URL = f'{ENDPOINT}/vision/v3.1/analyze'

headers = {
    'Ocp-Apim-Subscription-Key': API_KEY,
    'Content-Type': 'application/octet-stream'
}

# Initialize OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set up Google's Text-to-Speech credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "poem-0-matic-b9f292c4f549.json"

def clear_screen():
    system_name = platform.system().lower()
    if 'windows' in system_name:
        os.system('cls')
    else:
        os.system('clear')

def capture_image():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error initializing camera. Please check if it's connected and retry.")
        return False, None

    time.sleep(2)
    ret, frame = cam.read()
    cam.release()

    if ret:
        cv2.imshow('Captured Image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return ret, frame

def analyze_image_with_azure(url, image, params):
    _, img_encoded = cv2.imencode('.jpg', image)
    try:
        response = requests.post(url, headers=headers, data=img_encoded.tobytes(), params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
        print("Error Details:", response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
    return {}

def create_poem(description, dominant_color):
    context_info = f"The dominant color in the image is {dominant_color}."

    prompt = [
        {"role": "system", "content": "You are a poetic muse, able to capture the vibrant spirit and allure of Miami with your words."},
        {"role": "system", "content": "Remember to immerse the reader in Miami's unique atmosphere and culture."},
        {"role": "system", "content": "in the style of a sonnet"},
        {"role": "user", "content": f"Write a sonnet based on the image description: {description}. {context_info}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        temperature=0.6
    )

    choices = response.get('choices', [])
    if not choices:
        print("Unexpected response from OpenAI. Couldn't generate poem.")
        return ""

    return choices[0].get('message', {}).get('content', "")

def speak_text_google(text, pause_before=False):
    if pause_before:
        time.sleep(3)
        
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        name="en-US-Wavenet-F"
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)
    os.system("afplay output.mp3")  # for MacOS
    os.remove("output.mp3")

if __name__ == "__main__":
    try:
        clear_screen()

        if not API_KEY:
            print("Azure API key not found. Ensure the AZURE_API_KEY environment variable is set.")
            exit()

        if not openai.api_key:
            print("OpenAI API key not found. Ensure the OPENAI_API_KEY environment variable is set.")
            exit()

        success, captured_img = capture_image()
        if not success:
            print("Failed to grab frame from camera. Check camera availability.")
            exit()

        # Analyze image for dominant color
        analysis_params = {
            'visualFeatures': 'Color'
        }
        response_data = analyze_image_with_azure(ANALYZE_URL, captured_img, analysis_params)
        dominant_color = response_data.get('color', {}).get('dominantColorForeground', "")

        # Get description and tags
        description_data = analyze_image_with_azure(DESCRIBE_URL, captured_img, {})
        primary_description = description_data.get('description', {}).get('captions', [{}])[0].get('text', "")

        if primary_description:
            print("Image Description:", primary_description)
            poem = create_poem(primary_description, dominant_color)
            print("\n" + poem)
            
            # Speak out the poem
            speak_text_google(poem)

            sign_off = "Poem Created by Poem-O-Matic-AI by Mario The Maker for O'Miami festival"
            print(sign_off)
            
            # Speak out the sign off after a 3-second pause
            speak_text_google(sign_off, pause_before=True)
        else:
            print("Couldn't retrieve a description for the image.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
