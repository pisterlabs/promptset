import openai


def get_copy_edit(text: list, api_key: str, prompt: str):
    "Get copy edit from OpenAI API from a list of texts that need to be edited"
    openai.api_key = api_key
    responses = []
    for t in text:
        if t == "":
            responses.append("")
        else:
            responses.append(get_ai_response(t, prompt=prompt).strip())

    return responses


def get_ai_response(text, temperature=0.6, prompt=""):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"{prompt}'{text}'",
        temperature=temperature,
        max_tokens=int(len(text) * 1.3),
    )
    print(response.usage.total_tokens, " tokens used")
    return response.choices[0].text


# def generate_prompt(text):
#     return f"""Edit the following text, making the language more clear and convincing: "{text}"""


def get_tokens(text: str):
    return len(text) / 75 * 100
