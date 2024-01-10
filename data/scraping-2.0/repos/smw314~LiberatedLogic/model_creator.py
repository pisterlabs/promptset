import openai
import json

openai.api_key = "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"

def get_response(prompt, model="text-curie-002", max_tokens=150):
    response = openai.Completion.create(
        engine=model, #or "your_chosen_engine",
        prompt=prompt,
        max_tokens=max_tokens,
        n=5,
        stop=None,
        temperature=0.8,
    )

    return response.choices[0].text.strip()