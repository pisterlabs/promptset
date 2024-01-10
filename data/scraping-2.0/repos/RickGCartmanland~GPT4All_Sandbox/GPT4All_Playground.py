import openai
import json
import os
import time
import re

def sanitize_directory_name(name):
    return re.sub(r'\W+', '_', name)

openai.api_base = "http://localhost:4891/v1"
openai.api_key = "not needed for a local LLM"

# List of model names
model_ids = [
    'GPT4All Falcon', 'Groovy', 'ChatGPT-3.5 Turbo', 'ChatGPT-4', 'Snoozy', 
    'MPT Chat', 'Orca', 'Orca (Small)', 'Orca (Large)', 'Vicuna', 
    'Vicuna (large)', 'Wizard', 'Stable Vicuna', 'MPT Instruct', 
    'MPT Base', 'Nous Vicuna', 'Wizard Uncensored', 'Replit'
]

# Prompts
prompts = [
    "What are the building blocks of life according to science?",
    "What are the fundamental elements necessary for life according to chemistry?",
    "How do atoms contribute to the formation of life's building blocks?",
    "What role does DNA play as a building block of life?",
    "Can you explain the role of proteins as building blocks of life?",
    "How do carbohydrates serve as building blocks of life?",
    "What is the role of lipids in the composition of life's building blocks?",
    "How do nucleic acids function as building blocks of life?",
    "What are the differences between the building blocks of plant life and animal life?",
    "Can you discuss the building blocks of life found in extreme environments?",
    "How might our understanding of life's building blocks inform the search for extraterrestrial life?",
]

for prompt in prompts:
    # Create a directory for each prompt
    directory_name = sanitize_directory_name(prompt)
    os.makedirs(directory_name, exist_ok=True)

    for model in model_ids:
        for max_tokens in [100, 500, 2000]:  # short and long answers
            try:
                start_time = time.time()
                
                # Make the API request
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.28,
                    top_p=0.95,
                    n=1,
                    echo=True,
                    stream=False
                )

                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Print the model name and some details from the response
                print(f"Model: {model}")
                print(f"Max tokens: {max_tokens}")
                print(f"Model Used: {response['model']}")
                print(f"First Choice's Text: {response['choices'][0]['text']}")
                print(f"Execution time: {elapsed_time} seconds\n")

                # Convert the response to JSON and save it to a file
                filename = f"{model.replace(' ', '_')}_output_{max_tokens}_tokens.json"
                filepath = os.path.join(directory_name, filename)
                with open(filepath, 'w') as f:
                    f.write(json.dumps(response))
                    
            except Exception as e:
                print(f"An error occurred with model {model}: {e}")
