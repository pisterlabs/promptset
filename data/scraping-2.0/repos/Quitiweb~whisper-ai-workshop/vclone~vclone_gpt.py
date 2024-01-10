import os

import openai
from dotenv import load_dotenv
from playsound import playsound

from vclone_tools import text_to_vclone

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")

audio_path = "audio_vclone_v1.mp3"
chatgpt_model = "gpt-3.5-turbo"
chatgpt_system = "You are a helpful assistant on a conversation. Answer should be not too long. Be ironic and acid"

client = openai.Client()


def get_gpt_response(prompt):
    response = client.chat.completions.create(
        model=chatgpt_model,
        messages=[
            {"role": "system", "content": chatgpt_system},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def continuous_interaction():
    while True:
        # clear_output(wait=True)
        prompt = input("Enter your prompt (or type 'exit' to stop): ")
        if prompt.lower() == 'exit':
            break

        response_text = get_gpt_response(prompt)
        text_to_vclone(response_text, audio_path)
        playsound(audio_path)


# Example usage
continuous_interaction()
