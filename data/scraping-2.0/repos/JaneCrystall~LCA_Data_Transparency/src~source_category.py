import json
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from retrying import retry

load_dotenv()
openai_client = OpenAI()


@retry(wait_fixed=300, stop_max_attempt_number=5)
def query_func_calling(query: str):
    # Step 1: send the conversation and available functions to the model
    messages = [
        {
            "role": "user",
            "content": query,
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_source_category",
                "description": "Generating category for sources of LCA database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_category": {
                            "title": "source_category",
                            "description": "Generate category for sources of LCA database. The category must be one of the following: 'Article in periodical', 'Chapter in anthology', 'Monograph', 'Industry report', 'Standard/Directive', 'Environmental Product Declaration (EPD)','Patent', 'Statistical documents', 'Software or database', 'Personal communication','Direct measurement'.",
                            "type": "string",
                        },
                    },
                    "required": [
                        "source_category",
                    ],
                },
            },
        }
    ]
    response = openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        tools=tools,
        temperature=0.0,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    response_string = response_message.tool_calls[0].function.arguments
    return response_string

file_path = 'USLCI_source.xlsx'

# 读取Excel文件
df = pd.read_excel(file_path)
source_info = df['source_info']
for index, source in enumerate(source_info):
    try:
        source_category = query_func_calling(source)
        df.loc[df['source_info'] == source, 'source_category'] = source_category
        print(index + 1, source_category)
    except Exception as e:
        print(f"Error processing source '{source}': {e}")
        continue  # 继续下一个迭代


df.to_excel('USLCI_source_category.xlsx', index=False)