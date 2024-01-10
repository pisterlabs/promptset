import openai
from config import *


# Change "model" to whatever your desired model is. Such as GPT-4 if available
# Adjust temperature to either increase or decrease the randomness of ChatGPT's responses. 0.0-2.0
# Adjust max_tokens to increase or decrease the length of resposnes
def ChatGPT_conversation(conversation):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=conversation, max_tokens=200, temperature=1.0
    )
    conversation.append(
        {
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content,
        }
    )
    return conversation


def completed_assistant(
    prompt, conversation=[{"role": "system", "content": "You are a helpful assistant"}]
):
    conversation.append({"role": "user", "content": prompt})
    conversation = ChatGPT_conversation(conversation)
    return conversation[-1]["content"]
