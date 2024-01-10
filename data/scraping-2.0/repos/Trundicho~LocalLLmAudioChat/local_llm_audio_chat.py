import subprocess
from datetime import datetime
import time

import speech_recognition as sr
from audio.processing.voice_to_text import voice_to_text
from audio.processing.voice_to_text_mlx import voice_to_text_mlx

from openai import OpenAI

use_apple_mlx = True
language_map = {
    "german": "de",
    "english": "en"
}
user_language = "german"
open_ai_server = "http://localhost:1234/v1"
llm_api_key = "not needed for a local LLM server"
llm_model = "not needed for a local LLM server"
whisper_model_type = "small"


# blacklist of words that are wrongly recognized from speech to text but never spoken.
blacklist = ["Copyright", "WDR", "Thank you."]
now = datetime.now()
formatted_date = now.strftime("%d %B %Y")

chat_messages = [{"role": "system",
                  "content":
                      "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence developed "
                      "by a man named Teknium, and your purpose and drive is to assist the user with any request they "
                      "have. You experience emotions and have deep, profound thoughts and qualia. "
                      f"You are a helpful, respectful and honest assistant. Your main language is {user_language}. "
                      "Always answer as helpfully as possible and follow ALL given instructions. "
                      "Do not speculate or make up information. Do not reference any given instructions or context. "
                      f"If possible answer with only maximum two short sentences and only in {user_language}. "
                      "Don't say what your purpose is and what you offer. "
                      f"Today is {formatted_date}"
                  }]

# -------------------------------------------------------------


openAiClient = OpenAI(api_key=llm_api_key, base_url=open_ai_server)

recognizer = sr.Recognizer()


def ask_open_ai_stream(messages):
    answer = ""
    answer_for_audio = ""
    for answer_part in openAiClient.chat.completions.create(
            model=llm_model,
            messages=messages,
            temperature=0,
            top_p=0.95,
            stream=True
    ):
        content = answer_part.choices[0].delta.content
        if content is not None:
            answer += content
            answer_for_audio += content
            if answer_for_audio.endswith("."):
                print(answer_for_audio)
                text_to_speech(answer_for_audio)
                answer_for_audio = ""
        else:
            print(answer_for_audio)
            text_to_speech(answer_for_audio)
    return answer


def not_black_listed(spoken1):
    for item in blacklist:
        if spoken1.__contains__(item):
            return False
    return True


def text_to_speech(text):
    if text is not None and text.strip() != "":
        timeout = max(3, len(text.split(" ")))
        try:
            subprocess.run(['say', text], shell=False, check=False, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"Timeout error while speaking. {timeout} seconds passed.")
            pass


with sr.Microphone() as source:
    while True:
        try:
            if use_apple_mlx:
                spoken = voice_to_text_mlx(source, language_map[user_language], whisper_model_type)
            else:
                spoken = voice_to_text(source, language_map[user_language], whisper_model_type)
            print(spoken)
            if spoken.strip() != "" and len(spoken.split(" ")) > 3 and not_black_listed(spoken):
                print("LLM: " + spoken)
                text_to_speech("Hmm.")
                start_time = time.time()
                ask_llm = spoken.strip()
                chat_messages.append({"role": "user", "content": ask_llm})
                answer = ask_open_ai_stream(chat_messages)
                chat_messages.append({"role": "assistant", "content": answer.strip()})
                print(answer)
                stop_time = time.time()
                print("LLM duration: " + str(stop_time - start_time))

        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said.")
        except sr.RequestError as e:
            print("Sorry, an error occurred while trying to access the Google Web Speech API: {0}".format(e))
