import openai

# Set your OpenAI API key
api_key = 'YOUR_API_KEY'

# Initialize the OpenAI client
openai.api_key = api_key

# Define a prompt
prompt = "Translate the following English text to French: 'Hello, how are you?'"

# Generate a response using GPT-3
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=50  # You can adjust this based on the desired response length
)

# Extract the generated text from the response
generated_text = response.choices[0].text

print(generated_text)
