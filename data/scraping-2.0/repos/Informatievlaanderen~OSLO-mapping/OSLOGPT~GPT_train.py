import os
import openai
openai.api_type = "azure"
openai.api_base = "https://gpt92023.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = ''
# Load your training data
with open("training_dataset.txt", "r") as f:
    training_data = f.read()
    
print(training_data)

response = openai.ChatCompletion.create(
    engine="GPT43292023",  # "text-davinci-002",
    messages=[training_data],
    temperature=0.5,
    max_tokens=32000,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None)

# Print the response from the API
print(response.choices[0].text)
