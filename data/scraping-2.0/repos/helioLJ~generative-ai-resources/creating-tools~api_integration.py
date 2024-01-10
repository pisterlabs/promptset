from openai import OpenAI
client = OpenAI() # client will automatically look for OPENAI_API_KEY in env vars

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages= [
        {
            "role": "system",
            "content": """Voce é um gerador de produtos fictícios e voce deve gerar
            apenas o nome, sem nenhum tipo de descricao, dos produtos que o usuário solicitar."""
        },
        {
            "role": "user",
            "content": "Gere 5 produtos"
        }
    ]
)

print(response.choices[0].message.content)
print("\n")
print(response.usage)