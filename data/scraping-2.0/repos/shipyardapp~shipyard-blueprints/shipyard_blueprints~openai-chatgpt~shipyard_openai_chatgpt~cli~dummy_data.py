import os
import openai
import pandas as pd


def main():
    key = os.environ.get("CHATGPT_API_KEY")
    number_of_rows = os.environ.get("CHATGPT_NUMBER_OF_ROWS")
    column_names = os.environ.get("CHATGPT_COLUMNS")
    export_file = os.environ.get("CHATGPT_DESTINATION_FILE_NAME")

    openai.api_key = key

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Generate {number_of_rows} lines of the following fake data: {column_names}. The result should be presented in CSV format with a header row",
            }
        ],
    )

    print(completion.choices[0].message.content)

    with open(export_file, "w") as f:
        f.write(completion.choices[0].message.content)

    pd_csv = pd.read_csv(export_file)

    pd_csv.to_csv(export_file, index=False)
