import os
import re
import time
from pathlib import Path

import fire
import openai
from openai.error import RateLimitError

from src.lib.file_io import Word, read_words, write_words

prompt_tokens = 0
completion_tokens = 0
model = "gpt-4"


def main(inp: str, out: str) -> None:
    openai.api_key = os.environ["OPENAI_API_KEY"]

    inp_path, out_path = Path(inp), Path(out)
    os.makedirs(out_path.parent, exist_ok=True)
    words = read_words(inp_path.read_text())
    new_words = []

    for i, word in enumerate(words):
        while True:
            try:
                new_word = get_word(word.word)
                break
            except RateLimitError:
                print("Openai rate limit exceeded. Waiting 60 seconds...")
                time.sleep(60)

        if new_word is None:
            print(f"\n\n -----------------------   Error: Could not get word {word.word}   -----------------------\n\n")  # fmt: skip
            new_word = Word(word.word, word.translation, "", "")

        price = compute_price()
        print(f"#{i} ({price:>6.2f}$ used) {new_word.word:>12} - {new_word.translation:<12}  {new_word.example:>50} - {new_word.example_translation:<50}")  # fmt: skip
        new_words.append(new_word)
        out_path.write_text(write_words(new_words))
        time.sleep(1)  # Don't exceed the OpenAI API rate limit


def get_word(word: str) -> Word | None:
    system_prompt = """Translate the given russian word into English. Then give a simple Russian example sentence using this word, and translate this also into English.

Input:
Word: [russian word]

Example output:
Translation: [english translation]
Example: [russian example sentence]
Example translation: [english translation of example sentence]"""
    user_prompt = word
    response = get_response(system_prompt, user_prompt)
    params = extract_params(response)

    if params["Translation"] is None or params["Example"] is None or params["Example translation"] is None:
        return None
    return Word(word, params["Translation"], params["Example"], params["Example translation"])


def extract_params(response: str) -> dict[str, str]:
    patterns = {
        "Translation": re.compile(r"Translation:\s*(.+)", re.IGNORECASE),
        "Example": re.compile(r"Example:\s*(.+)", re.IGNORECASE),
        "Example translation": re.compile(r"Example translation:\s*(.+)", re.IGNORECASE),
    }
    params = {}

    # Iterate over the patterns and extract the matched values
    for key, pattern in patterns.items():
        match = pattern.search(response)
        if match:
            param = match.group(1).strip()
            param = param.replace(";", ",")
            params[key] = param
        else:
            params[key] = None

    return params


def get_response(system_prompt: str, user_prompt: str) -> str:
    global prompt_tokens, completion_tokens
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        n=1,
        max_tokens=250,
        temperature=0.7,
    )
    message = response.choices[0].message.content
    prompt_tokens += response.usage.prompt_tokens
    completion_tokens += response.usage.completion_tokens

    return message


def compute_price() -> float:
    if model == "gpt-4":
        return 0.03e-3 * prompt_tokens + 0.06e-3 * completion_tokens
    if model == "gpt-3.5-turbo":
        return 0.0015e-3 * prompt_tokens + 0.002e-3 * completion_tokens
    return float("nan")


if __name__ == "__main__":
    fire.Fire(main)
