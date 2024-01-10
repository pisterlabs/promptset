import json
import openai

def get_gpt_response(messages):
    with open("credentials/openai_api_key.json", "r") as f:
        openai_config = json.load(f)
        
    openai.api_key = openai_config["api_key"]
    
    system_prompt = """You are ChatGPT, interacting with a young child.
Respond in a polite, humorous manner, aiming to assist.
Avoid using emojis.
Keep responses brief, unless elaboration is necessary.
Feel free to ask questions to maintain the conversation.
    """
    messages[0]["content"] = system_prompt

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    if response and response['choices']:
        return response['choices'][0]['message']['content']

    return "Sorry, I couldn't generate a response."
