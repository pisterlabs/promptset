import json
import openai
import os
from api.utils.read_key import read_key

def send_request_to_openai(path_to_json_structure, save_filepath):
    """
    Send a request to OpenAI API using the provided JSONL structure.

    Parameters:
    - jsonl_structure (dict): The JSONL structure to send.

    Returns:
    - dict: The response from OpenAI API.
    """
    # Load resonator from file
    with open(path_to_json_structure, 'r') as f:
        jsonl_structure = json.load(f)


    response = openai.ChatCompletion.create(
        model=jsonl_structure["model"],
        messages=jsonl_structure["messages"]
    )
    with open(save_filepath, 'w', encoding="utf8") as f:
        json.dump(response, f, ensure_ascii=False)

# # Load resonator from file
# with open("output.jsonl", 'r') as f:
#     jsonl_structure = json.load(f)

# Send message using the resonator
read_key()
openai.api_key = os.getenv("OPENAI_API_KEY")
# # response = send_request_to_openai(jsonl_structure)
# # print(response)
