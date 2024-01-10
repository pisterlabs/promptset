import re
import random
import openai

# Set your OpenAI API key
openai.api_key = ''

# Path to the text file
text_file_path = r'C:\Users\haris\Desktop\combined.txt'



def extract_prompts_from_text(text):
    prompts = re.findall(r'"prompt": "(.*?)"', text)
    non_empty_prompts = [prompt for prompt in prompts if prompt.strip()]
    return non_empty_prompts

def generate_modulations(prompt):
    if prompt:
        modulations = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50,
            n=5
        )
        return [mod['text'].strip() for mod in modulations.choices]
    return []

with open(text_file_path, 'r', encoding='utf-8') as file:
    text_content = file.read()

non_empty_prompts = extract_prompts_from_text(text_content)

if non_empty_prompts:
    random_prompt = random.choice(non_empty_prompts)
    print(f"Prompt used: {random_prompt}")
    
    modulations = generate_modulations(random_prompt)
    
    if modulations:
        print("Possible modulations and Results:")
        for i, modulation in enumerate(modulations, start=1):
            print(f"{i}. {modulation}")
    else:
        print("No valid modulations generated.")
else:
    print("No non-empty prompts found in the file.")





