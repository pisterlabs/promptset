import os
import openai
from dotenv import load_dotenv
import requests
from elevenlabs import clone, generate, play, stream
from pydub import AudioSegment
import wave
import io
from elevenlabs.api.voice import Voice, VoiceSample

from voices import mirna1, mitchel1, mirna21, luis1

# Configure API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")

actual_voz = luis1

# Function to interact with GPT-4
def interact_with_gpt4(prompt, voice):
    # Implement your GPT-4 interaction logic here
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=256,
        n=1,
        stop=None,
        temperature=0.7
    )
    response_text = response.choices[0].text.strip()
    

    # Generate audio using cloned voice
    audio = generate(text=response_text, voice=voice)
    return audio

    # audio_stream = generate(
    # text=response_text,
    # stream=True, voice=voice
    # )
    # stream(audio_stream)
    # return audio_stream
       

# Function to save audio file
def save_audio_file(datos_audio, nombre_archivo, formato='mp3'):
    audio_io = io.BytesIO(datos_audio)
    audio = AudioSegment.from_file(audio_io)
    audio.export(nombre_archivo, format=formato)

# Main function
def main():
    # Set up voice cloning
    voice =  actual_voz

    while True:
        prompt = input("Enter your prompt (or type 'exit' to stop): ")
        if prompt.lower() == 'exit':
            break
        
        # Interact with GPT-4 and generate audio
        audio_file = interact_with_gpt4(prompt, voice)
        # print(audio_file)
        # audio_file = audio_file.decode('latin-1')
        # print(audio_file)
        # Save audio file
        nombre_archivo = 'audioclone.mp3'
        save_audio_file(audio_file, nombre_archivo)

        # # Save audio file
        # nombre_archivo = 'audioclone.mp3'
    
        # with open(audio_file, 'rb') as file:
        #     datos_audio = file.read()
        #     save_audio_file(datos_audio, nombre_archivo)
        #     print("Audio saved successfully.")

        # # Clean up the temporary audio file
        # os.remove(audio_file)
        # print("Temporary audio file deleted.")


if __name__ == "__main__":
    main()
