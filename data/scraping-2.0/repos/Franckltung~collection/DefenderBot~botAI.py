import os
import openai
import tiktoken
import constants

openai.api_key = os.getenv("OPENAI_API_KEY")

def craft_response(history, prompt):
    # Make sure length is not too long
    enc = tiktoken.encoding_for_model(constants.AI_MODEL)
    encoded_messages=enc.encode(history + prompt)
    if (len(encoded_messages) + constants.AI_MAX_TOKENS) > 4096:
        raise ValueError("GPT Message is too long!")

    response = openai.ChatCompletion.create(model=constants.AI_MODEL, messages=[
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "assistant",
            "content": history
        }
    ],
    temperature=1,
    max_tokens=constants.AI_MAX_TOKENS,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    return response
