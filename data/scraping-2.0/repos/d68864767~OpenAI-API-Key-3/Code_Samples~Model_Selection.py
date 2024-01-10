```python
import openai

# Set your OpenAI API key
openai.api_key = 'your-api-key'

# Define the prompt
prompt = "Translate the following English text to French: '{}'"
text = "Hello, world"

# Select the model
model = "text-davinci-002"  # For example, we are using the text-davinci-002 model

# Generate the response
response = openai.Completion.create(
  engine=model,
  prompt=prompt.format(text),
  max_tokens=60
)

print(response.choices[0].text.strip())
```
