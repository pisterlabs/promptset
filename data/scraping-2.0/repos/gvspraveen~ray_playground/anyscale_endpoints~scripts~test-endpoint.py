import openai
import os 
api_key = os.getenv('ANYSCALE_API_KEY')
api_base = "https://api.endpoints.anyscale.com/v1"
model = "meta-llama/Llama-2-7b-chat-hf"

response = openai.ChatCompletion.create(
        api_base=api_base,
        api_key=api_key,
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please explain Llama2 model architecture in plain english"}],
        temperature=0.9,
        max_tokens=4000
    )

print(response)
