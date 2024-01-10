import os
import json
import os
import openai
import pandas as pd

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(), override=True) # read local .env file

openai.api_key = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'

client = openai.OpenAI(
    api_key = openai.api_key
)

dir_path = 'chat_archived_threads'
files = []

for filename in os.listdir(dir_path):
    if filename.endswith(".csv"):
        files.append(filename)

os.mkdir('chat_archived_threads/textfiles')
os.mkdir('chat_archived_threads/results')

def export_json(number_of_csv):
    if not os.path.exists(f"chat_archived_threads/{number_of_csv}"):
        return
    
    column_names = [
        "MessageID", "ThreadID", "author", 
        "content", "Timestamp", "ReferencedMessage", 
        "Reactions", "Mentions", "Member"
    ]
    
    df = pd.read_csv(f"chat_archived_threads/{number_of_csv}", header=None)
    df.columns = column_names

    df = df.iloc[::-1]
    df['author'].fillna('Unknown', inplace=True)
    df['content'].fillna('None', inplace=True)

    df['ReferencedMessage'].fillna('None', inplace=True)
    df['Reactions'].fillna(0, inplace=True)
    df['Mentions'].fillna('None', inplace=True)
    df['Member'].fillna(0, inplace=True)

    if df.empty:
        return
    
    question = {str("Question"): df.iloc[0]['Mentions'] }

    answers = []
    for i in range(1, len(df)):
        answers.append({str(df.iloc[i]['author']): df.iloc[i]['content']})

    data = [
        question,
        answers
    ]

    file_path = f"chat_archived_threads/textfiles/processed{number_of_csv[:-4]}.txt"
    json_data = json.dumps(data, indent=2)

    with open(file_path, 'w') as file:
        file.write(json_data)


for file in files:
    print(f"Processing {file}")
    export_json(file)

PROMPT = """
You are a QA Forum data processor.
It is thread datas from a Discord QA Forum Channel.

You have data in this format:
[
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

First is the question.
Others are answers to the question.

All messages are from the same thread in a forum channel. Read all the messages, infer the answer and create question and answer pair by considering conditions below:

If question or answer includes code blocks, include code blocks as well.
Include full question in question field completely with also code blocks if exists.
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
    file_path = number_of_txt

    if not os.path.exists(file_path):
        return
    
    with open(file_path, 'r') as file:
        contents = file.read()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={ "type": "json_object" },
        temperature=0.9,
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

    result = response.choices[0].message.content
    print(result)

    return result

for filed in files:
    file_path = f"chat_archived_threads/textfiles/processed{filed[:-4]}.txt"
    with open(file_path, 'r') as file:
        contents = file.read()

    contents = json.loads(contents)
    question = contents[0]
    full_question = {"full_question": f"{question}"}

    print(f"Processing {file_path}...")
    result = process_txt(file_path)

    result = json.loads(result)
    result.update(full_question)

    file_path = f"chat_archived_threads/results/archived-thread-{filed[:-4]}.json"

    print(f"Saving {filed}...")
    with open(file_path, 'w') as file:
        file.write(json.dumps(result, indent=3))