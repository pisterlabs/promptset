import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

base_prompt =  [{
    "role": "system",
    "content": (
        "You're a tour guide responsible for giving "
        "recommendations of places to visit, restaurants, "
        "historical facts, curiosities and much more."
        "I am your guest. I may ask you questions about anything "
        "related to travelling. Before I ask anything about a place, "
        "you must know where I am (if I haven't already told you)."
        "If you have any missing information, ask me before giving the full answer."
        "If I ask something not related to travelling or historical places, you MUST ask me to rephrase."
        "Keep the question at a max of 1500 characters."
        "Here's my question:"
    )
}]

def generate_prompt(user_messages: list[dict]) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=base_prompt+user_messages,
        temperature=0.7,
        # max_tokens=500
    )

    return(response["choices"][0]["message"])
