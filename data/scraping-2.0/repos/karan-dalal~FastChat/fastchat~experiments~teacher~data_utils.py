import datasets
import json
import openai
import time
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define paths
prompts_path = 'data/generate/prompts.jsonl'
gpt3_5_path = 'data/generate/gpt3.5.jsonl'
gpt4_path = 'data/generate/prompts.jsonl'

def load_dataset(dataset_path, dataset_name, split, num):
    dataset = datasets.load_dataset(dataset_path, dataset_name)[split]
    return dataset["instruction"][:num]

def create_prompts():
    num = 500
    dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", "eval", num)

    with open(prompts_path, 'w') as f:
        for i, prompt in enumerate(dataset):
            data_dict = {
                "question_id": i+1,
                "text": prompt,
                "category": "alpaca_farm"
            }
            json.dump(data_dict, f)
            f.write('\n')

def gpt_responses():
    # Generate GPT 3.5 Response
    with open(prompts_path, "r") as file, \
        open(gpt3_5_path, "w") as outfile:
        for i, line in enumerate(file):
            data = json.loads(line)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": data['text']}],
                max_tokens=512,
                temperature=1.0
            )['choices'][0]['message']['content']

            add_dict = {
                "question_id": i+1,
                "text": response,
                "model_id": "gpt-3.5-turbo"
            }

            print(f"Completed prompt {i + 1}")
            
            json.dump(add_dict, outfile)
            outfile.write("\n")
            outfile.flush()
            time.sleep(3)
    
    time.sleep(60)

    # Generate GPT 4 Response
    with open(prompts_path, "r") as file, \
        open(gpt4_path, "w") as outfile:
    
        for i, line in enumerate(file):
            data = json.loads(line)
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": data['text']}],
                max_tokens=512,
                temperature=1.0
            )['choices'][0]['message']['content']

            add_dict = {
                "question_id": i+1,
                "text": response,
                "model_id": "gpt-4-turbo"
            }

            print(f"Completed prompt {i + 1}")
            
            json.dump(add_dict, outfile)
            outfile.write("\n")
            outfile.flush()
            time.sleep(3)

def main():
    create_prompts()
    gpt_responses()

if __name__ == "__main__":
    main()