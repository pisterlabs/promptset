from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv("../.env"))


def gptRequest(user, system, model='gpt-4', temperature=0.7):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user
            },
            {
                "role": "system",
                "content": system
            }
        ],
        model=model,
    )

    return response.choices[0].message.content
