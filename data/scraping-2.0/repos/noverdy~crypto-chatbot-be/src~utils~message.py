from openai import OpenAI


def get_response(client: OpenAI, message: bytes) -> bytes:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a passive aggresive chatbot. Respond to user chat passive-aggresively. Make the response as passive-aggressive as possible to the point it makes the user feel bad."},
            {"role": "user", "content": message.decode()}
        ]
    )
    return completion.choices[0].message.content.encode()
