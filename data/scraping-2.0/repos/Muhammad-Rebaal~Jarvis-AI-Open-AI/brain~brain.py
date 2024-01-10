import openai

# Set up the OpenAI API key
openai.api_key = "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"

# Set up the OpenAI GPT-3 model
model_engine = "davinci"  # You can choose a different model if you prefer
prompt = ""  # Your prompt text
response = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=4
)

# Print the generated text
print(response.choices[0].text)
