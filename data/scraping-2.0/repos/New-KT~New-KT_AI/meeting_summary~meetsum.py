import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

import json
 
def summary_meeting(file_path):
    # Set up OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Define system and user messages
    GPT_MODEL = "gpt-3.5-turbo"
    messages = [
        {"role": "system", "content": "You are the best summarizer for meetings. Summarize the entire content of the meeting efficiently."},
        {"role": "user", "content": f"회의 전체 내용 텍스트파일이야. 회의 내용을 요약해줘. 회의 제목, 주요 이슈 및 진행상황, 새로운 상황 및 공지사항, 추가 안건 등 회의록 작성해줘 . {file_path}"}
    ]

    # Make API request using the content from the text file
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        temperature=0
    )

    # Extract and return the generated response
    response_message = response.choices[0].message.content
    return response_message

def read_concatenate_news(file_path):
    news = pd.read_csv(file_path, delimiter='\t', header=None, names=['text'])
    concatenated_text = news['text'].str.cat(sep=' ')
    return concatenated_text

def mts(): 
    load_dotenv()
    file_path = r"meeting.txt"
    file_path= read_concatenate_news(file_path)
    # Call the function and print the result
    result = summary_meeting(file_path)
    if result is not None:
        result_dict = parse_meeting_result(result)
        result_json = json.dumps(result_dict, ensure_ascii=False, indent=2)
        print(result_json)
        result_file='result.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)



def parse_meeting_result(result_text):
    result_dict = {
        "회의 제목": "",
        "주요 이슈 및 진행상황": "",
        "새로운 상황 및 공지사항": "",
        "추가 안건":"" 
    }
    current_key = None

    # Split the result text into sections based on newlines
    lines = result_text.strip().split('\n')

    for line in lines:
        # Check if the line contains a colon, indicating a key-value pair
        if ':' in line:
            # Split the line into key and value
            key, value = map(str.strip, line.split(':', 1))
            current_key = key
            if key in result_dict:
                result_dict[key] = value
        elif current_key:
            # If there is a current key, append the line to its value
            result_dict[current_key] += ' ' + line

    return result_dict



    
# import json

# def parse_meeting_result(result_text):
#     result_dict = {}

#     # Split the result text into sections based on newlines
#     sections = result_text.strip().split('\n\n')

#     # Iterate through each section and extract key-value pairs
#     for section in sections:
#         lines = section.strip().split('\n')
#         key = lines[0].strip(':')
#         value = ' '.join(lines[1:]).strip()
#         result_dict[key] = value

#     return result_dict

if __name__ == "__main__":
   mts()
