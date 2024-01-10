import os

import openai

openai.api_key_path = "/home/kdhsimplepro/kdhsimplepro/kyunggi_highschool/CodingFestival2023/MEA/server/methods/chatgpt_api_key.txt"

def summarize_passage(passage):
    content = 'summarize the following passage in "one line":\n\n'
    content += passage

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": content}]
    )

    return response.choices[0].message.content


if __name__ == '__main__':

    with open("../db/passages/2022-09-1-21.txt", "r") as f:
        passage = f.readline()

    summary = summarize_passage(passage)
    print(summary)