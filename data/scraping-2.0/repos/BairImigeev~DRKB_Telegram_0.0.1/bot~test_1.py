import os
import openai
from telegram import Bot, Update

bot_token = "5013892743:AAFwYE56d_QdVCCTjAWXJi0LRSS1xnd2S2c"
bot = Bot(token=bot_token)
openai.api_key = os.environ['OPENAI_API_KEY']


def gpt3(prompt,
         model="text-davinci-002",
         temperature=0.7,
         max_tokens=1024,
         top_p=0.5,
         frequency_penalty=0,
         presence_penalty=0):

    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    return response.choices[0].text.strip()


def main():
    while True:
        prompt = input('--')
        text = gpt3(prompt, top_p=0.5)
        print(text)


if __name__ == "__main__":
    main()