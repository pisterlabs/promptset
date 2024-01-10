import openai
import os
import json
from dotenv import dotenv_values


#diff_output_からの読み込み
def read_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def generate_explain_error_question_in_the_code(line, explain_text, error_text, question):
    return {
        "line": line,
        "explain_text": explain_text,
        "error_text": error_text,
        "question": question
    }

def run_conversation_for_difference(difference, mistaken_code):
    line = difference["line"]
    # text_in_mistaken_code = difference["text_in_mistaken_code"]
    specific_line_in_mistaken_code = mistaken_code.split('\n')[line-1]  # 0-indexed, so subtract 1
    print(specific_line_in_mistaken_code)
    messages = [{"role": "user", "content": f"{mistaken_code} In the above program file, the operation and error contents in the normal case of `{specific_line_in_mistaken_code}` in the {line} line and the question text about the error are generated in Japanese."}]    
    functions = [{
        "name": "generate_explain_error_question_in_the_code",
        "description": "Create a question statement that elicits what the line of the program is doing, what the error is, and about the error.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "line": {
                        "type": "string",
                        "description":"Point out incorrect lines, e.g. 1,3,7"
                        },
                        "explain_text":{
                        "type":"string",
                        "description":"Explain in Japanese what the operation is in the correct answer."   
                        },
                        "error_text":{
                        "type":"string",
                        "description":"Explain in Japanese the contents of the line in which the error was made."   
                        },
                        "question":{
                        "type":"string",
                        "description":"Question text for asking about this mistake in japanese."   
                        }
                    },
                        "required": ["line", "explain_text", "error_text", "question"]
                }
            }

        ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",
    )
    response_data = {
        "id": response["id"],
        "object": "chat.completion",
        "created": response["created"],
        "model": "gpt-3.5-turbo-0613",
        "choices": response["choices"],
        "usage": response["usage"]
    }
    return response_data # Assuming program is a list of lines

config = dotenv_values("../.env")
openai.api_key = config["api_key"]

json_file_path = './generate/diff_output.json'
json_data = read_json_file(json_file_path)
mistaken_code = json_data["mistaken_code"]

all_responses = []
for difference in json_data["differences"]:
    response_data = run_conversation_for_difference(difference, mistaken_code)
    all_responses.append(response_data)

output_format = all_responses

print(output_format)

with open(f'./generate/all_error_text.json', 'w') as result_file:
    json.dump(output_format, result_file, ensure_ascii=False, indent=4)
