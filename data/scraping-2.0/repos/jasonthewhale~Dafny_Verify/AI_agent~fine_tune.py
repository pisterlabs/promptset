import tiktoken
import openai
import json


def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def read_json_file(file_path):
    # Open and read the JSON file
    with open( file_path, 'r') as file:
        # Parse the contents of the file into a Python list of dictionaries
        data = json.load(file)
    return data


# Check if all data within the dataset is less than 2048 tokens
datasets = read_json_file('../dataset/combined_dataset.json')
token_count = 0
for data in datasets:
    total_string = data['prompt'] + data['completion']
    num_tokens = num_tokens_from_string(total_string)
    token_count += num_tokens
    if num_tokens >= 2048:
        print(data['prompt'][:100])
        print('\n===========================\n\n')
print(f'Total amount of token is: {token_count}')
print(f'Esitimated cost of Ada is: {token_count * 0.0004 / 1000}')
print(f'Esitimated cost of Curie is: {token_count * 0.003 / 1000}')
print(f'Esitimated cost of Davinci is: {token_count * 0.03 / 1000}')