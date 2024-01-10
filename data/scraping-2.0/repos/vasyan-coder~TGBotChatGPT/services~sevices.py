import openai


def get_chatGPT_answer(message_text: str) -> str:
    response: dict = openai.Completion.create(
        model="text-davinci-003",
        prompt=message_text,
        temperature=1,
        max_tokens=1000,
        # top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0
    )

    return response["choices"][0]["text"]
