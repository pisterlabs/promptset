from dotenv import load_dotenv
load_dotenv()

import os
import openai

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

#model = "gpt-35-turbo"
model = "gpt-4"

prompt: str = "Write an introductory paragraph to explain Generative AI to the reader of this content." 
system_prompt: str = "Explain in detail to help student understand the concept.", 
assistant_prompt: str = None, 

messages = [
    {"role": "user", "content": f"{prompt}"},
    {"role": "system", "content": f"{system_prompt}"},
    {"role": "assistant", "content": f"{assistant_prompt}"}
]

openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_type = "azure"
openai.api_version = "2023-05-15" 
openai.api_base = f"https://{azure_endpoint}.openai.azure.com"

completion = openai.ChatCompletion.create(
    model = model, 
    engine = azure_deployment_name,
    messages = messages,
    temperature = 0.7
)

print(completion)
response = completion["choices"][0]["message"].content
print(response)
