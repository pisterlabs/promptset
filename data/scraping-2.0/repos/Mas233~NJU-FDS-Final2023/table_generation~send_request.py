import openai
from const import key

openai.api_key_path = key


def generate_request(prompt, model='text-davinci-003', temp=0, max_token=1024, timeout=2000):
    completion = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_token,
        n=1,
        stop=None,
        temperature=temp,
        timeout=timeout
    )
    response = completion.choices[0].text
    return response
