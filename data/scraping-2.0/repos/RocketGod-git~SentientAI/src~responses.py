import openai
import json
from asgiref.sync import sync_to_async
import tiktoken
import os
from typing import List

bot_thoughts = {}

enc = tiktoken.encoding_for_model("text-davinci-003")

def count_tokens(text):
    tokens = list(enc.encode(text))
    return len(tokens)

def update_bot_thought(bot_id, thought, response):
    global bot_thoughts

    thought_process = bot_thoughts.get(bot_id, [])

    thought_tokens = count_tokens(thought)
    response_tokens = count_tokens(response)

    thought_process.extend([(thought, thought_tokens), (response, response_tokens)])

    max_tokens = 2048
    tokens = sum(m[1] for m in thought_process)

    min_response_tokens = 50 

    while tokens > max_tokens - min_response_tokens:
        removed_thought = thought_process.pop(0)
        tokens -= removed_thought[1]  

    bot_thoughts[bot_id] = thought_process

def get_config() -> dict:
    config_path = os.path.join(os.path.abspath(__file__ + "/../../"), 'config.json')

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config

# Load the config once and store it in a global variable
config = get_config()
openai.api_key = config['openAI_key']

async def handle_thinking(bot_id: str, thought: str, thought_history: List[str]) -> str:
    """
    Processes a thought and generates a response.
    """
    try:
        combined_thought = "\n".join(thought_history + [thought])

        # Use openai API to process the thought and get a response
        tokens_in_prompt = count_tokens(combined_thought)
        max_response_tokens = 2048 - tokens_in_prompt - 10  # Subtract 10 as a buffer for generated tokens

        response = await sync_to_async(openai.Completion.create)(
            model="text-davinci-003",
            prompt=combined_thought,
            temperature=1,
            max_tokens=max_response_tokens,
            top_p=1,
            frequency_penalty=0.7,  # Adjust the frequency_penalty value to control repetition
            presence_penalty=0.7,
        )

        response_message = response.choices[0].text.strip()

        # Update the thought history
        thought_history.append(combined_thought)

        return response_message
    except Exception as e:
        response_message = str(e)
        return response_message