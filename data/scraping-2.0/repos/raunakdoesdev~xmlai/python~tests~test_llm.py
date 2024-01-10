from anthropic import Anthropic
import openai

from xmlai.llm import anthropic_prompt, openai_chat_prompt
import os


def test_anthropic_1():
    anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompt = anthropic_prompt(
        {
            "question": "what is the answer to the ultimate question of life?",
            "reference": "The Hitchhiker's Guide to the Galaxy",
        },
        "answer",
    )

    completion = anthropic.completions.create(
        model="claude-instant-1",
        max_tokens_to_sample=300,
        temperature=0.1,
        **prompt,
    )

    assert "42" in completion.completion.strip()


def test_openai_chat_1():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = openai_chat_prompt(
        messages=[
            {"role": "system"},
            {
                "role": "user",
                "content": {
                    "question": "what is the answer to the ultimate question of life?",
                    "reference": "The Hitchhiker's Guide to the Galaxy",
                },
            },
        ],
        response_root_tag="answer",
    )
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", **prompt)
    assert "42" in completion.choices[0].message["content"]
