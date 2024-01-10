import openai

# Set up your OpenAI API key
openai.api_key = "YOUR_API_KEY"

# Define your prompt
prompt = "Insert your prompt here"

# Generate the code
completion = openai.Completion.create(
  engine="davinci-codex",
  prompt=prompt,
  max_tokens=100
)

# Print the generated code
print(completion.choices[0].text.strip())