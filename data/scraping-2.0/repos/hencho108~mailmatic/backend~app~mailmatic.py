import argparse
import os

import openai

MAX_INPUT_LENGTH = 300


def validate_length(notes: str) -> bool:
    return len(notes) <= MAX_INPUT_LENGTH


def notes_to_email(notes: str) -> str:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    enriched_prompt = f"Convert my notes into a professional email. Notes: {notes}"

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=enriched_prompt,
        max_tokens=120,
        temperature=0.5,
    )
    email_text = response["choices"][0]["text"].lstrip()
    return email_text


def main(args):
    if validate_length(args.input):
        email = notes_to_email(args.input)
        print(email)
    else:
        raise ValueError(
            f"Input is too long. Must be under {MAX_INPUT_LENGTH} characters."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", "-n", type=str, required=False)
    args = parser.parse_args()
    main(args)
