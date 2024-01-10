import openai

openai.api_base = "http://localhost:8000/v1/aimodels/gpt4all/chat"
#openai.api_base = "https://api.openai.com/v1"

openai.api_key = "not needed for a local LLM"

# Set up the prompt and other parameters for the API request
prompt = "How are you?"

# model = "gpt-3.5-turbo"
#model = "mpt-7b-chat"
model = "ggml-gpt4all-l13b-snoozy.bin"

# Make the API request
response = openai.Completion.create(
    model=model,
    prompt=prompt,
    max_tokens=50,
    temperature=0.28,
    top_p=0.95,
    n=1,
    echo=True,
    stream=False
)

# Print the generated completion
print(response)
