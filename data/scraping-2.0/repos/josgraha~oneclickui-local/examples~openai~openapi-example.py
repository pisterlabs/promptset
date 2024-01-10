import openai

openai.api_type = "azure"
openai.api_key = "..."
openai.api_base = "http://localhost:5001/v1"
openai.api_version = "2023-05-15"

OPENAI_API_KEY = 'sk-111111111111111111111111111111111111111111111111'
MODEL = "TheBloke_Mistral-7B-OpenOrca-GPTQ"

# create a chat completion
chat_completion = openai.ChatCompletion.create(
    deployment_id="deployment-name",
    model=MODEL,
    messages=[{"role": "user", "content": "4 + 4 = ?"}],
)

# print the completion
print(chat_completion.choices[0].message.content)
