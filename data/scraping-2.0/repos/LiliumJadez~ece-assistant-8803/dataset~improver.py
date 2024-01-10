import json
import openai

def optimize_dataset(dataset):
    """Optimize the given dataset using ChatGPT-3.5-turbo."""
    optimized_dataset = []

    for entry in dataset:
        messages = [
            {"role": "system", "content": "Suppose you are a helpful academic advisory chatbot, you manage to answer the question about georgia tech ECE programme."},
            {"role": "user", "content": f"Instruction: {entry['instruction']}\nInput: {entry['input']}\nOutput: {entry['output']}\n\nRewrite the instruction, input, and output to be more natural and flowing, while keeping the main information intact:"}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150
        )

        last_message = response['choices'][0]['message']['content']
        optimized_text = last_message.strip()
        optimized_parts = optimized_text.split('\n')

        if len(optimized_parts) >= 3:
            optimized_dataset.append({
                "instruction": optimized_parts[0].replace("Instruction: ", "").strip(),
                "input": optimized_parts[1].replace("Input: ", "").strip(),
                "output": optimized_parts[2].replace("Output: ", "").strip()
            })
        else:
            print(f"Warning: Response format not as expected for entry: {entry}")

    return optimized_dataset

# Replace 'your_api_key' with your actual OpenAI API key
openai.api_key = 'your_api_key'

# Load your dataset (assuming the file is named 'dataset.json')
with open('dataset.json', 'r') as file:
    dataset = json.load(file)

# Optimize the dataset
optimized_dataset = optimize_dataset(dataset)

# Save the optimized dataset
with open('optimized_dataset.json', 'w') as file:
    json.dump(optimized_dataset, file, indent=4)

print("Dataset optimized and saved as 'optimized_dataset.json'")
