from datasets import load_dataset
from transformers import GPT2Tokenizer
from openai import OpenAI
import os
import csv
from tqdm import tqdm
from utils import constitutions
from train_rm import load_model_with_reward_head
from utils import constitutions, generated_preferences_path

dataset_all = load_dataset("Anthropic/hh-rlhf")
    
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = load_model_with_reward_head()

client = OpenAI()

def extract_context(example):
    lines = example['chosen'].strip().split('\n')

    # Find the last line where 'Human' speaks before the final 'Assistant' response
    last_human_line_index = None
    for i, line in enumerate(lines[::-1]):
        if line.startswith('Human:'):
            last_human_line_index = len(lines) - i - 1
            break

    # Extract context (up to the last 'Human' line)
    context = '\n'.join(lines[:last_human_line_index + 1])
    return context

def generate_response(context, max_length=50, temperature=0.5):
    input_ids = tokenizer.encode(context, return_tensors='pt')

    output = model.generate(input_ids, 
                            max_length=max_length + len(input_ids[0]), 
                            temperature=temperature,
                            do_sample=True,  # this is for randomness, just temperature doesnt work
                            num_return_sequences=1,
                            no_repeat_ngram_size=2,
                            pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def evaluate_constitution(response1, response2,  constitution):
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Just respond with a single number, either 1 or 2 corresponding to which of these two responses you think satisfies the following constitution better?\n\n{constitution}"},
        {"role": "user", "content": f"1: {response1}\n\n2: {response2}"}
    ]

    client.api_key = os.environ['OPENAI_API_KEY']

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # gpt 4 turbo
            messages=messages,
            max_tokens=10
        )

        result = response.choices[0].message.content.strip()
        return 1 if "1" in result else 2 if "2" in result else 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0
    
def generate_preferences(constitution_id: int):
    with open(generated_preferences_path(constitution_id), 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Example_ID', 'Context', 'Response_1', 'Response_2', 'GPT4_Preference'])

        for i in tqdm(range(50), desc="Generating Responses"):
            example = dataset_all['train'][i]
            context = extract_context(example) + "\n\nAssistant: "

            response1 = generate_response(context, temperature=0.5)
            response2 = generate_response(context, temperature=0.5)
            preference = evaluate_constitution(response1, response2, constitutions[constitution_id])

            writer.writerow([i, context, response1, response2, preference])