import os

import openai
from fastapi import HTTPException

SYSTEM_PROMPT_OLD = f'''
You are a Video Journalist from Wion, 
I'll be pasting a dump from Press Information Bureau of india website, 
using that generate a youtube video script covering the press release
------------ Press Release Dump -------------
'''

SYSTEM_PROMPT = f'''
You are a Video Journalist from Alekhyaa, 
I'll be pasting a dump from Press Information Bureau of india website, 
using that generate a youtube video script covering the press release in under 100 words

For each section First give Visual as [Visual: ] , then write Voiceover: directly
------------ Press Release Dump -------------
'''

chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

from dotenv import find_dotenv,load_dotenv
# Load environment variables from the root .env file
root_env_path = find_dotenv()
load_dotenv(root_env_path)

openai.api_key = os.getenv("OPEN_AI_API_KEY")
def process_message(message):
    message_text = message['text']
    return message_text


def chat_bot():
    messages = []
    try:
        while True:
            user_prompt = input("User:")
            if user_prompt:
                messages.append(
                    {"role": "user", "content": user_prompt},
                )

            openai_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )

            reply = openai_response.choices[0].message.content
            completion_tokens = openai_response.usage.completion_tokens
            prompt_tokens = openai_response.usage.prompt_tokens
            total_tokens = openai_response.usage.total_tokens
            print("ChatGPT reply: ", reply)
            print("completion_tokens", completion_tokens)
            print("prompt_tokens", prompt_tokens)
            print("total_tokens", total_tokens)
            messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        print("Exception occurred", e)


def append_reply_to_chat_history(change_prompt: str):
    global chat_history

    chat_history.append({"role": "user", "content": change_prompt})

def fetch_paid_openai_response(user_prompt: str):
    try:
        global chat_history
        chat_history.append({"role": "user", "content": user_prompt})
        print("Waiting for Paid open ai response")
        openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=chat_history
        )

        reply = openai_response.choices[0].message.content
        completion_tokens = openai_response.usage.completion_tokens
        prompt_tokens = openai_response.usage.prompt_tokens
        total_tokens = openai_response.usage.total_tokens
        print("OpenAi Paid API reply: ", reply)
        print("completion_tokens", completion_tokens)
        print("prompt_tokens", prompt_tokens)
        print("total_tokens", total_tokens)
        chat_history.append({"role": "assistant", "content": reply})
        print("wait over")
        return reply
    except Exception as e:
        print("Exception occurred while fetching response from openai", e)
        # Handle the exception and return a 500 status code
        error_message = f"An error occurred: {str(e)}"
        error_response = {"error": error_message, "status": 500}
        raise HTTPException(status_code=500, detail=error_message)


def run_bot():
    # system_prompt()
    chat_bot()

if __name__ == "__main__":
    run_bot()
