import os
import json
import openai
import datetime

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = 'gpt-3.5-turbo'
DATA_FILE_PATH = './data/The History of Jazz.txt'
OUTPUT_FILE_PATH = './data/'
INSTRUCTION_SYS = "you are a language expert who teaches English"
INSTRUCTION_USER = "Given the text, can you provide the definition, usage, and an example sentence for the advanced vocabularies and slang?"
openai.api_key = OPENAI_API_KEY

def get_results(input_messages):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=input_messages,
            temperature=0,
            max_tokens=3500
        )
        return response.choices[0].message['content']
    except Exception as e:  # Catch specific exception
        print(f"Error occurred: {e}")
        return 'Error'

def build_messages(text, max_length=2048, step=2044):
    messages =[]
    for i in range(0, len(text), step):
        chunk = text[i:i + max_length]
        prompts = [
            {"role": "system", "content": INSTRUCTION_SYS},
            {"role": "user", "content": INSTRUCTION_USER},
            {"role": "user", "content": f"{chunk}"}
        ]
        messages.append(prompts)
    return messages

def main():
    with open(DATA_FILE_PATH, 'r') as file:
        input_text = file.read()
    messages = build_messages(input_text)

    current_time = datetime.datetime.now()
    timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    print(f'opening file result_summarization_{timestamp_str}.txt')
    with open(f'{OUTPUT_FILE_PATH}result_summarization__{timestamp_str}.txt', 'w', encoding='utf-8') as f:
        for idx, message in enumerate(messages):
            result = get_results(message)
            print(result)
            f.write(result)
    if f:
        f.close()

if __name__ == "__main__":
    main()
    print("Program Completed!")
