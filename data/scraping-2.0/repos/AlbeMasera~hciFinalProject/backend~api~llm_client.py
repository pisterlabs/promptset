import openai
from backend.settings import OPENAI_API_KEY
from openai import OpenAI

client = OpenAI()


def paraphrase(system_content):
    openai.api_key = OPENAI_API_KEY
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "system", "content": system_content}],
        response_format={"type": "json_object"},
    )
    answer = completion.choices[0].message
    print("Answer: ", answer.content)
    answer = answer.content.replace("\n", " ")
    return answer
