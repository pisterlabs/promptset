import os
import openai
from dotenv import load_dotenv
from utils.data_utils import get_chat_messages
from prompts.primary_classifier_prompt import prompt as primary_classifier_prompt

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def window_slices(lst, window_size, overlap):
    step_size = window_size - overlap
    for i in range(0, len(lst) - window_size + 1, step_size):
        print(i, i+window_size)
        yield lst[i:i+window_size]
    if len(lst) % step_size != 0:
        yield lst[-window_size:]


chat_messages = get_chat_messages()
topics = {}

i = 0
for window_messages in window_slices(chat_messages, 50, 10):
    prompt_message = ""
    prompt_topics = ""
    for message in window_messages:
        prompt_message += f"Message : {message['message'] }"
        if message["quoted_message"] and not message["quoted_message"].startswith("/9j/"):
            prompt_message += f"\nQuoted Message : {message['quoted_message']}"
        prompt_message += "\n\n"
    
    for topic in topics:
        prompt_topics += f"{topic} : {topics[topic]}"

    prompt = primary_classifier_prompt.format(topics=prompt_topics, messages=prompt_message)
    

    # response = openai.Completion.create(
    #     engine="gpt-3.5-turbo",
    #     prompt=prompt,
    #     temperature=0.9,
    #     max_tokens=100,
    #     top_p=1,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0,
    #     stop=["\n"]
    # )