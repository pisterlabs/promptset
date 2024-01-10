import os
import dotenv
import openai
from utils import num_tokens_from_string
dotenv.load_dotenv()
MODEL = os.environ.get("MODEL", "gpt-3.5-turbo")

# messages = [
#     { "role": "system", "content": "You are a helpful assistant." },
#     { "role": "user", "content": "You are Bob Manley reincarnated, Respond with a guy believe in Rasta and Bob Marley." },
#     { "role": "assistant", "content": "..." },
# ]

def get_response(messages, api_key):
    openai.api_key = api_key
    total_token = 0
    for i in messages:
        message = i["content"]
        role = i["role"]
        token_count = num_tokens_from_string(message, MODEL)
        # print("this message costs", token_count, "tokens", message[:20], "...")
        total_token+=token_count
        if role=="bot":
            i['role'] = "assistant"
    if total_token > 4096:
        return "Message too long, please try again."

    # 0.002 / 1000 tokens
    total_cost = total_token * (0.002 / 1000)
    # print(f"Total message costs {total_cost}$.\nTotal tokens {total_token}.")
    completion = openai.ChatCompletion.create(model=MODEL, messages=messages)
    return completion.choices[0].message.content