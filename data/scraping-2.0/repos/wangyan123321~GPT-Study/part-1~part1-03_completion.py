import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
COMPLETION_MODEL = "text-davinci-003"

prompt = """
请以父亲和老师的口吻，分别跟孩子讲一则大灰狼和小红帽的故事。
"""

def get_response(prompt):
    response = openai.Completion.create(
        engine=COMPLETION_MODEL,
        prompt=prompt,
        temperature=1.0,
        max_tokens=1024,
        n=1,
        stop=None,
    )
    return response.choices[0].text

print(get_response(prompt))