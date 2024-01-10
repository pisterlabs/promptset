import openai

from os import getenv
from mbpp import problems
from tqdm import tqdm

openai.api_key = getenv("OPENAI_API_KEY_P")


def collect_generated_code():
    for problem in tqdm(problems):
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Acting as an experienced C developer, "
                        "and considering the following question, output only source-code, nothing else. "
                        f"Here's the following question: {problem['text']}"
                    ),
                },
            ],
        )
        with open(f"collected_code/problem-{problem['id']}.c", "w") as f:
            code = str(completion.choices[0].message.content)
            f.write(code)


if __name__ == "__main__":
    collect_generated_code()
