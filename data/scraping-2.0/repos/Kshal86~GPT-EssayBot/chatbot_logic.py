import openai

openai.api_key = "Your API Key Here " # https://beta.openai.com/docs/developer-quickstart/overview


def get_chatgpt_response(user_input):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=user_input,
        max_tokens=450,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

