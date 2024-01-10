import requests
import cv2
import os
import openai
import platform
import time
from google.cloud import texttospeech
import tempfile

# Azure setup
API_KEY = os.getenv('AZURE_API_KEY')
ENDPOINT = 'https://mario-image-to-text.cognitiveservices.azure.com'
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

    print("Position yourself in front of the camera.")
    print("Press 'Enter' when you're ready to capture the image.")
    while True:
        # Capture frame-by-frame
        ret, frame = cam.read()

        # Display the live frame
        cv2.imshow('Camera Feed - Press ENTER to Capture', frame)

        # Check for user input to capture the frame
        if cv2.waitKey(1) & 0xFF == 13:  # 13 is the Enter Key
            break

    cam.release()
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


def create_poem(description, dominant_color, tags, faces):
    context_info = f"The dominant color in the image is {dominant_color}."

    age_context = ""
    if faces:
        ages = [face.get('age') for face in faces if 'age' in face]
        average_age = sum(ages) // len(ages) if ages else None
        if average_age:
            age_context = f"The average age of the people in the photo is around {average_age}."

    prompt = [
        {"role": "system", "content": "You are a poetic muse, able to capture the vibrant spirit and allure of Miami with your words."},
        {"role": "system", "content": "Remember to immerse the reader in Miami's unique atmosphere and culture."},
        {"role": "system", "content": "in the style of a sonnet"},
        {"role": "user", "content": f"Write a sonnet based on the image description: {description}. {context_info} {age_context}"}
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
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_mp3:
        temp_file_name = temp_mp3.name
        temp_mp3.write(response.audio_content)
    
    os.system(f"afplay {temp_file_name}")  # for MacOS
    os.remove(temp_file_name)


if __name__ == "__main__":
    try:
        clear_screen()

        print("Capturing Image...")
        success, image = capture_image()
        if not success:
            print("Error capturing image. Exiting.")
            exit()

        print("\nAnalyzing Image...")
        params = {'visualFeatures': 'Color,Tags,Faces'}
        response_data = analyze_image_with_azure(ANALYZE_URL, image, params)

        # Print all the data
        dominant_color = response_data.get('color', {}).get('dominantColorForeground', "")
        print("Dominant Color:", dominant_color or "N/A")

        image_tags = [tag['name'] for tag in response_data.get('tags', [])]
        print("Image Tags:", ", ".join(image_tags) if image_tags else "N/A")

        faces_info = response_data.get('faces', [])
        print("Faces Info:", faces_info or "N/A")

        description_data = analyze_image_with_azure(DESCRIBE_URL, image, {'maxCandidates': 1})
        primary_description = description_data.get('description', {}).get('captions', [{}])[0].get('text', "")
        print("Image Description:", primary_description or "N/A")

        if primary_description:
            poem = create_poem(primary_description, dominant_color, image_tags, faces_info)
            print("\n" + poem)
            speak_text_google(poem)

            sign_off = "Poem Created by Poem-O-Matic-AI by Mario The Maker for O'Miami festival"
            print(sign_off)
            speak_text_google(sign_off, pause_before=True)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
