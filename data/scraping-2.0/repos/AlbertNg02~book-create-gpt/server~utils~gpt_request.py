import openai


def get_response(input):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a book writer"},
            {"role": "user", "content": input}
        ]
    )
    return completion["choices"][0]["message"]["content"]
