from openai import OpenAI

def main_openai(messages, openai_key, model="gpt-4", temperature=0.0):
    """
    Executes requests to OpenAI.
    """
    client = OpenAI(api_key = openai_key)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return completion.choices[0].message.content

