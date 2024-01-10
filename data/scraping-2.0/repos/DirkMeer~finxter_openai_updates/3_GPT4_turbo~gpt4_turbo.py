from decouple import config
from openai import OpenAI
from pathlib import Path

client = OpenAI(api_key=config("OPENAI_API_KEY"))
path_to_book = Path(__file__).parent / "book.txt"


def book_summary_GPT(file_path):
    with open(file_path, "r", encoding="utf8") as file:
        book = file.read()

    messages = [
        {
            "role": "system",
            "content": "You are a book-summarizing AI. You will receive a book as the query and you will return a summary which is not too long and summarizes the important main points and happenings of the book. Make sure to use only the text provided for your summary, and not any other knowledge you may have.",
        },
        {
            "role": "user",
            "content": book[:30_000],
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
    )

    content = response.choices[0].message.content
    # write the output to summary.txt
    with open("summary.txt", "w") as file:
        file.write(content)

    print(content)
    return content


book_summary_GPT(path_to_book)
