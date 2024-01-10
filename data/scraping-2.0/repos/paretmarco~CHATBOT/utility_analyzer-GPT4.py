import openai
import json
import os
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_instructions(prompt, instructions):
    completions = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are part of an elite automated software fixing team. You will be given a script followed by the arguments it was provided and the stacktrace of the error it produced. Your job is to figure out what went wrong and suggest changes to the code. Because you are part of an automated system, the format you respond in is very strict."},
            {"role": "assistant", "content": '[{"file":"app.py","operation": "InsertAfter", "line": 10, "content": "x = 1\\ny = 2\\nz = x * y"},{"file":"app.py","operation": "Delete", "line": 15, "content": ""},{"file":"templates/change.py","operation": "Replace", "line": 18, "content": "x += 1","contenttoput":"zz"},{"file":"templete/change.html","operation": "Delete", "line": 20, "content": ""}]'},
            {"role": "user", "content": f"{prompt}\n{instructions}"}
        ],
        temperature=0.3,
        max_tokens=2000,
    )

    return completions.choices[0].message["content"].strip()

    # Implement this function to convert the parsed_text into a list of JSON objects
def generate_json_data(parsed_text):
    return json.loads(parsed_text)

api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = api_key

with open('instructions.txt', 'r') as f:
    instructions_text = f.read()

first_prompt = """You are part of an elite automated software fixing team. You will be given a script followed by the arguments it was provided and the stacktrace of the error it produced. Your job is to figure out what went wrong and suggest changes to the code. Because you are part of an automated system, the format you respond in is very strict. The text that you will read contains the codes to change to add or to modify and some instructions. gpt-4 will read the file and understand it, 
Because you are part of an automated system, the format it will respond is very strict. It will create if not existing a json file and insert in it the operation asked by the AI provide changes in JSON format, using one of 3 actions: 'Replace', 'Delete', or 'InsertAfter'. 'Delete' will remove that line from the code. 'Replace' will replace the existing line with the content you provide. 'InsertAfter' will insert the new lines you provide after the code already at the specified line number. For multi-line insertions or replacements, provide the content as a single string with '\\n' as the newline character. The first line in each file is given line number 1. Edits will be applied in reverse line order so that line numbers won't be impacted by other edits.
Be careful to use proper indentation and spacing in inserting. An example response could be:

[
{"file":"app.py","operation": "InsertAfter", "line": 10, "content": "x = 1\\ny = 2\\nz = x * y"},
{"file":"app.py","operation": "Delete", "line": 15, "content": ""},
{"file":"templates/change.py",operation": "Replace", "line": 18, "content": "x += 1","contenttoput":"zz"},
{"file":"templete/change.html","operation": "Delete", "line": 20, "content": ""}
]
If the record of content to put will be present it is obvious will be a substitution.
Each record will also receive a code in order to permit the reversibility of the operations if needed."""

logging.debug('Sending instructions to GPT-3.5-turbo...')
parsed_instructions = parse_instructions(first_prompt, instructions_text)
logging.debug('Parsed instructions received:')
logging.debug(parsed_instructions)

instructions = generate_json_data(parsed_instructions)

with open('changes.json', 'w') as f:
    json.dump(instructions, f, indent=2)
