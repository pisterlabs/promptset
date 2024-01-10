import os 
import openai

os.environ['API_KEY'] = '<your api key goes here>'
openai.api_key = os.environ.get('API_KEY')

prompt = "What are the public hours for the bank's city branch?"

response = openai.completions.create(
    model = "text-davinci-003",
    prompt = prompt,
    max_tokens = 75,
)

print("Model: ", response.model)
print("Created: ", response.created)
print("ID: ", response.id)

response_print = response.choices[0].text.strip()
print(response_print)
