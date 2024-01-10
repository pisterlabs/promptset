import openai
from openai import OpenAI


def askPPLX(prompt):
    client = OpenAI(
        base_url="https://api.perplexity.ai",
        api_key="pplx-",
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to "
                "engage in a helpful, detailed, polite conversation with a user."
            ),
        },
        {
            "role": "user",
            "content": (
                prompt
            ),
        },
    ]

    # demo chat completion without streaming
    response = client.chat.completions.create(
        model="pplx-70b-online",
        messages=messages,
    )
    print(response.choices[0].message.content)

with open("easy.txt", "r") as file:
    for line in file:
        askPPLX(line.strip())

