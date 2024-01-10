import openai
import random
import string
import os
import traceback
import json


def get_chatgpt_response(model_name, messages, temperature=0.7):
    try:
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        prompt = messages[-1]["content"].rstrip(string.punctuation)

        response = openai.ChatCompletion.create(
            model=model_name, messages=messages, temperature=temperature
        )

        print("Raw API response:", json.dumps(response, indent=2))

        if response["choices"]:
            reply = response["choices"][0]["message"]["content"]
            return reply
        else:
            return "I'm sorry, but I couldn't generate a response for that question."

    except Exception as e:
        error_message = str(e)
        print(f"Error in get_chatgpt_response: {error_message}")
        print(traceback.format_exc())

        return f"Error occurred: {error_message}"
