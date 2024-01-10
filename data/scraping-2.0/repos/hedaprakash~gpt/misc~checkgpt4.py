import openai

# Initialize the OpenAI API with your API key
openai.api_key = 'sk-b0zIcjasdfasdfasdfsdfsdfsdfsdffsdf'

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
]

try:
    response = openai.ChatCompletion.create(
      model="gpt-4",  # Replace with the specific GPT-4 engine ID if different
      messages=messages
    )
    model_name = response['model']  # Extracting the model name from the response
    print(f"Response from {model_name}: {response['choices'][0]['message']['content']}")
except Exception as e:
    print(f"An error occurred: {e}")
