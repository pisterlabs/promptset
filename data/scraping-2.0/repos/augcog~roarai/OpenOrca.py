from datasets import load_dataset
import openai
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
dataset = load_dataset("Open-Orca/OpenOrca", split = "train")


# Define a container for the new responses
updated_responses = []

# Create an empty data dictionary with the required columns
data = {
    'id': [],
    'system_prompt': [],
    'question': [],
    'response': []
} 
n=2
selected_dataset=dataset[:n]
for i in range(n):
    for key in selected_dataset:
        system_prompt=selected_dataset['system_prompt'][i]
        question = selected_dataset['question'][i]
        # Combine or use the columns as needed for your API prompt
        api_prompt = system_prompt + " " + question

    # Generate samples using the API
        response = openai.ChatCompletion.create(
            model = "gpt-4", 
            prompt=api_prompt,
            max_tokens=50  # Adjust based on your needs
        )

    # Extract response and append to your container
        data_sample = response.choices[0].text.strip()
        data['id'].append(selected_dataset['id'][i])
        data['system_prompt'].append(system_prompt)
        data['question'].append(question)
        data['response'].append(data_sample)

   

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('updated_dataset.csv', index=False)



