import os
import openai


# set API key with the environment variable
openai.api_key = os.environ["OPENAI_API_KEY"]


def call_chatgpt(prompt):
    # call ChatGPT
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{'role': 'user', 'content': prompt}]
    )

    # extract content from response
    content = response.choices[0]['message']['content']
    return content


if __name__ == '__main__':
    prompt = input('prompt: ')
    content = call_chatgpt(prompt)
    print(content)
