""" downloader """

import json
import os
import sys
from pathlib import Path

import click
import openai
from dotenv import load_dotenv


DATA_DIR = Path(__file__).parent / "responses"
DATA_DIR.mkdir(exist_ok=True)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@click.command()
@click.option("--prompt", prompt="Enter description", help="Image description")
@click.option("--number", prompt="Enter number of images", help="Number of images")
def main(prompt: str, number: str) -> None:
    """ main func """

    while len(prompt) < 3:
        prompt = input("Too short! Enter description: ")

    while not number.isdigit():
        number = input("Not a number! Enter number: ")

    try:
        response = openai.Image.create(
            prompt=prompt[:256],
            n=int(number),
            size="256x256",
            response_format="b64_json",
        )
    except openai.OpenAIError as excp:
        print(f"HTTP status: {excp.http_status}; {excp.error}")
        sys.exit(-1)

    file_path = DATA_DIR / f'{hash(prompt) % 10_000}_{response["created"]}.json'  # type: ignore

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(response, file)


if __name__ == "__main__":
    main()
