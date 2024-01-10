import openai
import json
import os
import random
from dotenv import load_dotenv

load_dotenv()

# Get the API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Constants
INPUT_DIR = os.path.join("..", "data", "formatted_interviews")
OUTPUT_DIR = os.path.join("..", "data", "refined_data")

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

def is_short_content(content):
    return len(content.split()) <= 3

def generate_contextual_conversation(question, answer):
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Generate a 3-question, 3-answer conversation that leads up to and ends with the exact same Question and Answer as the one provided. Ensure the tone for all of the answers you generate matches the original answer. Start each question with 'Question:' and each answer with 'Answer:'. Make sure there is a newline between each question and answer and make sure the conversation makes sense."},
            {"role": "user", "content": f"Question: {question}, Answer: {answer}"}
        ]
    )
    return completion.choices[0].message['content']

def parse_contextual_conversation(conversation_text):
    lines = conversation_text.split("\n")
    messages = []
    user_count = 0
    assistant_count = 0

    for line in lines:
        if "Question:" in line:
            role = "user"
            user_count += 1
        elif "Answer:" in line:
            role = "assistant"
            assistant_count += 1
        else:
            continue

        content = line.split(":")[1].strip()
        messages.append({"role": role, "content": content})

        if user_count == 2 and assistant_count == 2:
            break

    return messages

def augment_prompt(question, answer):
    augmented_prompts = []
    num_prompts = random.choice([1, 2])  # Randomly choose between 1 and 2 prompts
    
    for _ in range(num_prompts):
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. You will be provided a prompt and a response. Rework the prompt to make it standalone and contextually appropriate for the given response while still maintaining some of the contextual details from the original prompt. Return only the reworked prompt."},
                {"role": "user", "content": f"Prompt: {question}, Response: {answer}"}
            ]
        )
        reworked_question = completion.choices[0].message['content']
        augmented_prompts.append(reworked_question)
    
    return augmented_prompts

def refine_dataset(filename):
    with open(os.path.join(INPUT_DIR, filename + ".json"), 'r') as f:
        data = json.load(f)

    refined_data = []
    for conversation in data:
        user_content = conversation['messages'][0]['content']
        assistant_content = conversation['messages'][1]['content']

        if is_short_content(user_content) or is_short_content(assistant_content):
            continue

        # Create a contextual conversation 15% of the time
        # if random.random() < 0.15:
        #     contextual_conversation_text = generate_contextual_conversation(user_content, assistant_content)
        #     contextual_messages = parse_contextual_conversation(contextual_conversation_text)
        #     contextual_messages.append({"role": "user", "content": user_content})
        #     contextual_messages.append({"role": "assistant", "content": assistant_content})
        #     refined_data.append({"messages": contextual_messages})
        # else:
        augmented_prompts = augment_prompt(user_content, assistant_content)
        for prompt in augmented_prompts:
            refined_data.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant_content}
                ]
            })

    with open(os.path.join(OUTPUT_DIR, filename + ".json"), 'w') as f:
        json.dump(refined_data, f, indent=4)

if __name__ == "__main__":
    filename = input("Enter the name of the file (without .json extension): ")
    refine_dataset(filename)
    print(f"Refined dataset saved to {os.path.join(OUTPUT_DIR, filename + '.json')}")
