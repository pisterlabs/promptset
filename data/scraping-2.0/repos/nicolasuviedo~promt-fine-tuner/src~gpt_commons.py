import openai
import json
from datetime import datetime

OPEN_API_KEY = "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"

openai.api_key = OPEN_API_KEY


def generate_text(
    prompt, model="gpt-3.5-turbo", max_tokens=150, n=1, temperature=1.0, meta=None
):
    if model == "gpt-3.5-turbo":
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": f"{prompt}"},],
            max_tokens=max_tokens,
            n=n,
            temperature=temperature,
        )
        generated_text = response.choices[0]["message"]["content"]
    else:
        response = openai.Completion.create(
            engine=str(model),
            prompt=prompt,
            max_tokens=max_tokens,
            n=n,
            temperature=temperature,
        )
        generated_text = response.choices[0].text.strip()

    # Store the input and output information
    api_call_details = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt_type": meta,
        "input": {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "n": n,
            "temperature": temperature,
        },
        "output": [generated_text],
    }

    # Append the information to a file as a dictionary
    with open("api_call_records.json", "a") as records_file:
        records_file.write(json.dumps(api_call_details) + "\n")

    return generated_text
