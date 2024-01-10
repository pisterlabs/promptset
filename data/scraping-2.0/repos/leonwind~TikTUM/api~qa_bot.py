from typing import Optional

import openai
import os

# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "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"


def ask_question(question: str,
                 extra_information: Optional[str] = None,
                 temperature: float = 0,
                 max_tokens: int = 100,
                 top_p: float = 1,
                 frequency_penalty: float = 0,
                 presence_penalty: float = 0) -> Optional[str]:
    prompt = "Answer this question"
    if extra_information:
        prompt += f" based on the following information: {extra_information}\n"

    prompt += f" Q: {question}\n"
    # Make request to GPT-3
    print(prompt)
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=["Answer this question"]
    )
    answer = response.get('choices')[0].get('text').strip()
    if answer and answer.startswith('A:'):
        answer = answer[2:].strip()

    return answer
