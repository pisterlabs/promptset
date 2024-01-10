import os
import openai
import pandas as pd


def main():
    key = os.environ.get("CHATGPT_API_KEY")
    original_text = os.environ.get("CHATGPT_FILE")
    export_file = os.environ.get("CHATGPT_DESTINATION_FILE_NAME")

    df = pd.read_csv(original_text)
    ten_records = df.head(10)

    openai.api_key = key

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Create a data dictionary for this data: {ten_records}",
            }
        ],
    )

    print(completion.choices[0].message.content)

    with open(export_file, "w") as f:
        f.write(completion.choices[0].message.content)
