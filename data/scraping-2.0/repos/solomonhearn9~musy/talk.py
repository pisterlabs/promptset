import openai
from flask import requests
import re

# Function to read API keys or other sensitive data from a file
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Loading API keys and predefined conversations from files
api_key = open_file('openaiapikey.txt')
chatbot1 = open_file('chatbot1.txt')
# You might want to manage this conversation array dynamically or keep it if it's needed as is
conversation1 = []

# Function to communicate with GPT-3.5 and get the response
def chatgpt(api_key, conversation, chatbot, user_input, temperature=0.9, frequency_penalty=0.2, presence_penalty=0):
    openai.api_key = api_key
    conversation.append({"role": "user", "content": user_input})
    messages_input = conversation.copy()
    prompt = [{"role": "system", "content": chatbot}]
    messages_input.insert(0, prompt[0])
    
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        messages=messages_input
    )

    chat_response = completion['choices'][0]['message']['content']
    conversation.append({"role": "assistant", "content": chat_response})
    
    return chat_response

# Function to transcribe audio using OpenAI's Whisper API
def transcribe_file(file_path):
   with open("temp_audio.wav", "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            response_format="json"
        )
   return {'text': transcript['text']}

def get_bot_response(user_message):
    response = chatgpt(api_key, conversation1, chatbot1, user_message)
    user_message_without_generate_image = re.sub(r'(Response:|Narration:|Image: generate_image:.*|)', '', response).strip()
    return user_message_without_generate_image

def convert_text_to_speech(text):
    ELEVENLABS_API_ENDPOINT = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    ELEVENLABS_API_KEY = open_file('elabapikey.txt')
    
    # Replace {voice_id} with the actual ID of 'myra' voice
    voice_id = "your_myra_voice_id"
    tts_url = ELEVENLABS_API_ENDPOINT.format(voice_id=voice_id)

    headers = {
        "Authorization": f"Bearer {ELEVENLABS_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    response = requests.post(tts_url, json=data, headers=headers, stream=True)
    
    if response.status_code == 200:
        with open("response_audio.mp3", "wb") as audio_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    audio_file.write(chunk)
        return "response_audio.mp3"
    else:
        print("Error:", response.status_code, response.text)
        return None