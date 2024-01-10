import openai
import requests
from PIL import Image
from gtts import gTTS
import base64
from io import BytesIO

# Replace with your OpenAI API key
openai.api_key = "your_openai_api_key"

def generate_text(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    generated_text = response.choices[0].text.strip()
    return generated_text

def generate_image(prompt):
    # Replace `your_openai_api_key` with your actual API key
    api_key = "your_openai_api_key"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": "image-alpha-001", "prompt": prompt, "n": 1, "size": "1024x1024"}
    response = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=data)

    if response.status_code == 200:
        img_url = response.json()["data"][0]["url"]
        img_response = requests.get(img_url)
        img = Image.open(BytesIO(img_response.content))
        img_buffer = BytesIO()
        img.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        return img_base64
    else:
        print("Error generating image:", response.status_code)
        return None

def text_to_speech(text):
    tts = gTTS(text, lang="en")
    speech_buffer = BytesIO()
    tts.write_to_fp(speech_buffer)
    speech_base64 = base64.b64encode(speech_buffer.getvalue()).decode("utf-8")
    return speech_base64

if __name__ == "__main__":
    prompt = input("Enter your prompt: ")

    generated_text = generate_text(prompt)
    print("Generated Text:", generated_text)

    image_base64 = generate_image(generated_text)
    print("Generated Image (Base64):", image_base64)

    speech_base64 = text_to_speech(generated_text)
    print("Generated Speech (Base64):", speech_base64)

