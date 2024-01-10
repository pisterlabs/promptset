import openai


def get_response(rude, key):
    openai.api_key = key
    prompt = "Convert the following prompt into a paragraph with a business formal tone and polite word choice: "
    full = prompt + rude
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=full,
        temperature=0.4,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.4
    )
    return response['choices'][0]['text']
