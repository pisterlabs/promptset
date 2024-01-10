import openai
import os


def send_prompt_openai(client, instructions, prompt, engine="gpt-3.5-turbo", temp=0.5, max_tokens=100, top_p=1, frequency_penalty=0, presence_penalty=0, timeout=None):
    respuesta = client.chat.completions.create(
        model=engine,
        messages=[
            {'role':'system', 'content':instructions},
            {'role': 'user', 'content': prompt}
        ],
        temperature=temp,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        timeout = timeout
    )
    return respuesta.choices[0].message.content


if __name__ == "__main__":
    pass
