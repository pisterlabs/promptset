# Description: This script uses the OpenAI GPT-3.5-turbo model to chat with the user.
# The user can speak to the chatbot and the chatbot will speak back.
# This script uses the speech_to_text_whisper.py script to convert speech to text.
# This script uses the text_to_speech_gc.py script to convert text to speech.
# Bryan Randell 2023


from speech_to_text_whisper import main_audio_to_text
from text_to_speech_gc import text_to_audio, voices_dict
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# speech_to_text = main_audio_to_text()

def gpt_chat_response(messages, model_engine="gpt-4"):
    chat = openai.ChatCompletion.create(
        model=model_engine,
        messages=messages,
    )

    response = chat["choices"][0]["message"]["content"]
    role = chat["choices"][0]["message"]["role"]
    return role, response

def main_speech_input_speech_output():
    model_engine = "gpt-3.5-turbo"  # Replace with "gpt-4" once available
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    while True:
        role = "user"
        input_message = main_audio_to_text()
        print(f"User: {input_message}")
        if not input_message:
            break
        # if input_message == "stop":
        #     break
        messages.append({"role": role, "content": input_message})
        role, response = gpt_chat_response(messages, model_engine)
        if len(response) > 0 and len(response) < 200:
            text_to_audio(response)
        else:
            text_to_audio("Too long to speak, read the answer")
        print(f"Bot {role}: {response}")
        messages.append({"role": role, "content": response})

if __name__ == "__main__":
    main_speech_input_speech_output()
