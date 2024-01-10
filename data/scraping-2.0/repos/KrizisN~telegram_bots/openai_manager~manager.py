import openai

ENGINE = "text-davinci-003"
MAX_TOKENS = 4000


def get_answer(question):
    completion = openai.Completion.create(
        engine=ENGINE,
        prompt=question,
        max_tokens=MAX_TOKENS,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    response = completion.choices[0]

    return response.text if response is not None else None
