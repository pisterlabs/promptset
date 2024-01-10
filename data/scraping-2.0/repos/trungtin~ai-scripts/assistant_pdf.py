#! python

import argparse

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

client = OpenAI()
assistant_model = "gpt-3.5-turbo-1106"


def main(pdf_files: list[str]):
    created_files = []
    for pdf_file in pdf_files:
        with open(pdf_file, "rb") as f:
            print(f"Uploading file {pdf_file} to OpenAI Assistant API.")
            content = f.read()
            file_name = pdf_file.split("/")[-1]
            # upload new file
            uploaded_file = client.files.create(
                file=(file_name, content, "application/pdf"),
                purpose="assistants",
            )
            print(f"Uploaded file {file_name} successfully to OpenAI Assistant API.")
            created_files.append(uploaded_file.id)

    # Create a new assistant
    assistant = client.beta.assistants.create(
        name="pdf_assistant",
        file_ids=created_files,
        model=assistant_model,
        tools=[{"type": "retrieval"}],
    )

    print(
        f"Created assistant: https://platform.openai.com/playground?assistant={assistant.id}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take pdf file(s) then create a OpenAI assistant with then."
    )
    parser.add_argument("pdf_files", type=str, nargs="+", help="path to the pdf files")

    args = parser.parse_args()
    main(args.pdf_files)
