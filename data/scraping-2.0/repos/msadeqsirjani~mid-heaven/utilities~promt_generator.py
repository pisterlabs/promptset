import openai
from settings import OPEN_AI_API_TOKEN


def generate(data) -> str:
    prompts = [f"{value}" for key, value in data.items()]
    prompt = "describe a house with a given features in a way that human can understand: " + ", ".join(prompts)
    prompt = prompt + " photographic"

    openai.api_key = OPEN_AI_API_TOKEN

    model_engine = "text-davinci-003"

    completion = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    response = completion.choices[0].text

    return response
