import csv
import os
import openai
import pandas as pd
import random
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def get_data_sample(filepath, num_lines=20, random_sample=False):
    """
    Get the header and `num_lines` of data from the csv
    """

    with open(filepath, "r") as f:
        lines = f.readlines()

    header = lines[0]
    data = None

    if random_sample:
        num_lines = min(num_lines, len(lines))
        indices = random.sample(range(1, len(lines)), num_lines-1)
        data = "".join([lines[i] for i in indices])
    else:
        end = min(num_lines+1, len(lines))
        data = "".join(lines[1: end])

    return header, data


def get_chat_topic(filename, header, data):
    """
    Use the filename and data to get a GPT-generated header for the file
    """

    prompt = f"""
    Here's an excerpt from a file titled "{filename}".
    As you can tell, the column names within the csv don't perfectly reflect the actual content of the data.
    Generate a set of headers more representative of the data to replace the current header.
    Output the answer as a string and using the same delimiter as shown in the header, in the order of
    the columns that they should label, with no other output.

    Header: {header}

    Data: {data}
    """

    print("prompt:", prompt)

    client = openai.OpenAI(api_key = OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
        {"role": "system",
        "content": prompt}
        ],
        temperature=0,
        max_tokens=512
    )
    return response.choices[0].message.content


def generate_csv(gpt_headers, filepath):
    # Specify the path to your CSV file
    f = open(filepath, "r")
    lines = f.readlines()
    lines[0] = gpt_headers
    if gpt_headers[-1] != "\n":
        lines[0] += "\n"

    directory, filename = os.path.split(filepath)
    gpt_filepath = os.path.join(directory, "GPT HEADER " + filename)
    with open(gpt_filepath, "w") as f:
        f.writelines(lines)

    return gpt_filepath

def generate_gpt_header(filepath):
    """
    Given a csv file, use GPT to generate a more representative set of headers
    and save the results in a new file
    """

    header, data = get_data_sample(filepath, random_sample=True)
    filename = os.path.basename(filepath)
    gpt_headers = get_chat_topic(filename, header, data)

    print("gpt headers:", gpt_headers)

    return generate_csv(gpt_headers, filepath)

if __name__ == '__main__':
    generate_gpt_header("../available_datasets/target_type.csv")

