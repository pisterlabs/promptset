import openai

openai.api_key = '***'
prompt = "Translate the following English text to French: 'Hello, how are you?'"

response = openai.Completion.create(
  engine='text-davinci-003',  # Specify the language model to use
  prompt=prompt,
  max_tokens=100  # Set the desired length of the response
)

output = response.choices[0].text.strip()  # Extract the generated response from the API response
print(output)
