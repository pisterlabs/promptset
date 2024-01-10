from openai import OpenAI
client = OpenAI()

def categorizeProduct(product_name, valid_categories):
    system_prompt = f"""
    Voce é um categorizador de produtos.
    Voce deve escolher uma categoria da lista abaixo:

    #### Lista de categorias válidas
    {valid_categories}
    #### Exemplo
    bola de tenis
    Esportes
    """

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": system_prompt
        },
        {
        "role": "user",
        "content": product_name
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    #   n=5
    )
    print(response.choices[0].message.content)



while True:
    print("Digite o nome do produto! ")
    product_name = input()
    categorizeProduct(product_name, "Beleza, Esportes, Entretenimento, Outros")