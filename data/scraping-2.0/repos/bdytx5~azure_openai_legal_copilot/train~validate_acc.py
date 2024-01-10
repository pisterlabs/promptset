
import openai
import json
import wandb
import time

# Initialize wandb
wandb.init(project="openai-model-evaluation")

# Set up OpenAI credentials
openai.api_key = "" 
openai.api_base = ""
openai.api_type = 'azure'
openai.api_version = '2023-09-15-preview'

# Set DEBUG_MODE to True to print inputs and expected outputs without running inference
DEBUG_MODE = False

def run_inference_and_calculate_accuracy(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    total_entries = 0
    correct_predictions = 0

    for line in lines:
        time.sleep(0.1)  # To avoid hitting API rate limits
        data = json.loads(line)
        messages = data["messages"]
        sys_message = next((m['content'] for m in messages if m['role'] == 'system'), None)
        user_message = next((m['content'] for m in messages if m['role'] == 'user'), None)
        expected_answer = next((m['content'] for m in messages if m['role'] == 'assistant'), '').strip().lower()

        if DEBUG_MODE:
            prompt = [{"role": "system", "content": sys_message}, {"role": "user", "content": user_message}]
            print(f"Input: {str(prompt)}")
            print(f"Expected Output: {expected_answer}")
            print("-" * 20)
        else:
            prompt = [{"role": "system", "content": sys_message}, {"role": "user", "content": user_message}]
            response = openai.ChatCompletion.create(
                engine="deploy_stock",  # Replace with your custom model name
                messages=prompt
            )
            model_answer = response['choices'][0]['message']['content'].strip().lower()

            total_entries += 1
            if model_answer == expected_answer:
                correct_predictions += 1

    if not DEBUG_MODE:
        accuracy = (correct_predictions / total_entries) * 100
        wandb.log({"accuracy": accuracy})
        return accuracy

# Replace 'your_file_path.jsonl' with the path to your JSONL data file
file_path = '../data/legalbench_subsets_val.jsonl'

if DEBUG_MODE:
    run_inference_and_calculate_accuracy(file_path)
else:
    accuracy = run_inference_and_calculate_accuracy(file_path)
    print(f"Model Accuracy: {accuracy}%")
    wandb.finish()
