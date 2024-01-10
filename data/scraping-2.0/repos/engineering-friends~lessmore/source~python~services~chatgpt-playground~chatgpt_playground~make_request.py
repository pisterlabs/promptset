import openai
import pytest


@pytest.mark.slow
def make_request(prompt: str, **prompt_kwargs) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt.format(**(prompt_kwargs or {})),
            },
        ],
    )

    return response.choices[0].message.content


def test():
    assert make_request("say foo") == "foo"


if __name__ == "__main__":
    test()
