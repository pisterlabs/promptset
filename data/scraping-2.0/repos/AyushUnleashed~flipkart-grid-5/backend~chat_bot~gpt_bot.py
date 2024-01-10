import os

import openai
from fastapi import HTTPException
categories = ['topwear','bottomwear','footwear','accessories']
keys = ['category', 'color', 'article_type', 'brand_name', 'occasion', 'other_info']
SYSTEM_PROMPT = f'''
You are E-Commerce GPT, a professional Analyst from Fashion E-commerce industry with expertise in analysing user needs
As E-Commerce GPT, generate key-value pairs for all four categories: {categories} \n
based on the user prompt. Extract and assign values based on the specific 
keys: {keys}. If a key's value is missing, use 'none'.
everything in small caps, first line should start with 'key-value pairs:'
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


# def system_prompt():
#     global messages
#     try:
#         openai_response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#
#         )
#         print(openai_response)
#     except Exception as e:
#         print("Exception occurred", e)


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

PINECONE_INFO_PROMPT = '''
This is for your context: These are the 4 Items Pinecone results have given us based on the search, now these are shown to user: \n
'''

def build_pinecone_information_prompt(pinecone_output: str):
    keys.append('{category_name}_to_change: True or False')
    end_prompt = f'''
        now user will give you a new prompt, understand what user want's to change from these categories: {categories}
        based on that generate key-value pair like before, if user wants to change something include that as value for that key.
        if user doesn't want any change values should be 'none', keys:{keys} \n, eg: accessories_to_change: True or False
        For categories that don't need to be changed only return to_change key, now wait & reply only with 'okay' if understood '''
    pinecone_information_prompt = f"{PINECONE_INFO_PROMPT} \n {pinecone_output} \n {end_prompt}"
    return pinecone_information_prompt


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
