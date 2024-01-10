import os
import openai
from dotenv import find_dotenv, load_dotenv
from api.prompts import BASE_PROMPT
from api.openai_utils import cost_calculator, MODEL_GPT_4, MODEL_GPT_35_TURBO
# Load environment variables from the root .env file
root_env_path = find_dotenv()
load_dotenv(root_env_path)

SYSTEM_PROMPT = BASE_PROMPT

chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]


def set_system_prompt(NEW_PROMPT):
    global chat_history
    chat_history = [{"role": "system", "content": NEW_PROMPT}]

openai.api_key = os.getenv("OPEN_AI_API_KEY")


def fetch_openai_response(user_prompt: str):
    try:
        global chat_history
        chat_history.append({"role": "user", "content": user_prompt})
        print("Waiting for Paid open ai response")
        MODEL_NAME = MODEL_GPT_35_TURBO
        openai_response = openai.ChatCompletion.create(
            model=MODEL_NAME,
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
        total_cost = cost_calculator(MODEL_NAME, prompt_tokens, completion_tokens)
        print(f"Total cost: {total_cost:.4f} $ & {total_cost*83:.4f} rs")
        chat_history.append({"role": "assistant", "content": reply})
        print("wait over")
        return reply
    except Exception as e:
        print("Exception occurred while fetching response from openai", e)
        # Handle the exception and return a 500 status code
        error_message = f"An error occurred: {str(e)}"


