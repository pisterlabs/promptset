# python -m vllm.entrypoints.openai.api_server --model models/TheBloke_SUS-Chat-34B-AWQ


from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(model="models/TheBloke_SUS-Chat-34B-AWQ",
                                      prompt="San Francisco is a")
print("\nCompletion result:", completion)



client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
chat_response = client.chat.completions.create(
    model="models/TheBloke_SUS-Chat-34B-AWQ",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
)
print("\nChat response:", chat_response)




