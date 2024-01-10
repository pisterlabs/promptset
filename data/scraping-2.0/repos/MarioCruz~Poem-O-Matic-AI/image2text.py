import requests
import cv2
import os
import openai
import time
import platform

# Azure setup
API_KEY = os.getenv('AZURE_API_KEY')
ENDPOINT = 'https://image-to-text.cognitiveservices.azure.com/'
DESCRIBE_URL = f'{ENDPOINT}/vision/v3.1/describe'
ANALYZE_URL = f'{ENDPOINT}/vision/v3.1/analyze'

headers = {
    'Ocp-Apim-Subscription-Key': API_KEY,
    'Content-Type': 'application/octet-stream'
}

# Initialize OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def clear_screen():
    system_name = platform.system().lower()
    if 'windows' in system_name:
        os.system('cls')
    else:
        os.system('clear')

def capture_image():
    cam = cv2.VideoCapture(0)
    time.sleep(2)
    ret, frame = cam.read()
    cam.release()
    return ret, frame

def send_image_to_azure(url, image, params=None):
    _, img_encoded = cv2.imencode('.jpg', image)
    try:
        response = requests.post(url, headers=headers, data=img_encoded.tobytes(), params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
    return {}

def create_poem(description, tags, emotions):
    tag_string = ', '.join(tags[:5])
    most_prominent_emotions = [max(emotion, key=emotion.get) for emotion in emotions]
    emotion_string = ', '.join(most_prominent_emotions)
    context_info = ""
    if tags:
        context_info += f" The image has elements like {tag_string}."
    if emotions:
        context_info += f" The faces in the image express emotions such as {emotion_string}."

    prompt = [
        {"role": "system", "content": "You are a poet from Miami, flordai and all your poems are sonnets. Make it miami "},
        {"role": "user", "content": f"Write a poem inspired by the image description: {description}.{context_info}"}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt
    )
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    clear_screen()
    success, captured_img = capture_image()
    if not success:
        print("Failed to grab frame from camera. Check camera availability.")
        exit()

    # Get description and tags
    response_data = send_image_to_azure(DESCRIBE_URL, captured_img)
    primary_description = response_data['description']['captions'][0]['text'] if 'description' in response_data else ""
    tags = response_data['description']['tags'] if 'description' in response_data else []

    # Print description and tags
    if primary_description:
        print("Image Description:", primary_description)
    if tags:
        print("Tags:", ", ".join(tags))

    # Get emotions of faces
    face_params = {"visualFeatures": "Faces"}
    face_data = send_image_to_azure(ANALYZE_URL, captured_img, params=face_params)
    emotions = []

    if 'faces' in face_data:
        for face in face_data['faces']:
            if 'faceAttributes' in face and 'emotion' in face['faceAttributes']:
                emotions.append(face['faceAttributes']['emotion'])

    # Print emotions
    most_prominent_emotions = [max(emotion, key=emotion.get) for emotion in emotions]
    print("Emotions:", ", ".join(most_prominent_emotions))

    if primary_description:
        poem = create_poem(primary_description, tags, emotions)
        print("\nGenerated Poem:\n", poem)
    else:
        print("Couldn't retrieve a description for the image.")
