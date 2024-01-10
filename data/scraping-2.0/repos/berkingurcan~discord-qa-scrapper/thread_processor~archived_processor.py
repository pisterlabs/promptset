import json
import os
import openai
import time
import pandas as pd

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(), override=True) # read local .env file

openai.api_key = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'

client = openai.OpenAI(
    api_key = openai.api_key
)

def export_json(number_of_csv):
    df = pd.read_csv(f"archived_threads/{number_of_csv}.csv")
    df = df.iloc[::-1]

    thread_name = df.iloc[0]['thread_name']
    question = {str(df.iloc[0]['author']): df.iloc[0]['content'] }

    answers = []
    for i in range(1, len(df)):
        answers.append({str(df.iloc[i]['author']): df.iloc[i]['content']})

    data = [
        thread_name,
        question,
        answers
    ]

    file_path = f"archived_threads/textfiles/processed{number_of_csv}.txt"
    json_data = json.dumps(data, indent=2)

    with open(file_path, 'w') as file:
        file.write(json_data)

    try:
        with open(file_path, 'r') as file:
            contents = file.read()
    except FileNotFoundError:
        print(f"File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    


os.mkdir("archived_threads/textfiles")
os.mkdir("archived_threads/results")

def count_files_in_folder(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

folder_path = "archived_threads"
length_of_folder = count_files_in_folder(folder_path)

for i in range(length_of_folder):
    export_json(i)


PROMPT = """
You are a QA Forum data processor.
It is thread datas from a Discord QA Forum Channel.

You have data in this format:
[
    "Thread Name",
    {
        "Author": "Question"
    },
    [
        {
            "Author": "Answer"
        },
        {
            "Author": "Answer"
        },
        {
            "Author": "Answer"
        }
        ...
    ]
]

First is the name of the thread.
Second(inside the curly bracket) is the question.
Others are answers to the question.

All messages are from the same thread in a forum channel. Read all the messages, infer the answer and create question and answer pair by considering conditions below:

Question need to be complete and include thread name.
If question or answer includes code blocks, include code blocks as well.
Include thread name and full question in question field completely with also code blocks if exists.
Please do not summarize the answer. Evaulate and explain the answer as detailed as possible with all necessary information which is also has code blocks.

Infer, evaulate and create Full Question: Detailed Answer Pair
Return a valid JSON as the final result, if there is no answer in the messages, return null. Thank you is not an answer, this data will be used for training so please remove unnecessary data.
Give me a JSON file with the following format in markdown format:
```json
{
"question": "The question",
"answer": "The answer" or None
}
```
"""

def process_txt(number_of_txt):
    file_path = f"archived_threads/textfiles/processed{number_of_txt}.txt"
    with open(file_path, 'r') as file:
        contents = file.read()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={ "type": "json_object" },
        temperature=0.8,
        messages=[
            {
                "role": "system",
                "content": PROMPT
            },
            {
                "role": "user",
                "content": str(contents)
            }
        ]
    )
    print(response)
    result = response.choices[0].message.content
    print(result)

    return result

def count_files_in_folder(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

folder_path = "archived_threads/textfiles"
length_of_folder = count_files_in_folder(folder_path)
    
for i in range(length_of_folder):
    file_path = f"archived_threads/textfiles/processed{i}.txt"
    with open(file_path, 'r') as file:
        contents = file.read()

    contents = json.loads(contents)
    thread_name = contents[0]
    question = contents[1]

    full_question = {"full_question": f"{thread_name}\n{question}"}

    print(f"Processing {i}...")
    result = process_txt(i)

    if result is None:
        continue

    result = json.loads(result)
    result.update(full_question)

    file_path = f"archived_threads/results/archived-thread-{i}.json"

    print(f"Saving {i}...")
    with open(file_path, 'w') as file:
        file.write(json.dumps(result, indent=4))

    time.sleep(3)
