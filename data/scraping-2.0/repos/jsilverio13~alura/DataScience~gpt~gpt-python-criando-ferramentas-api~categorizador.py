import os
import openai
import dotenv

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.system('clear')

def categoriza_produto(nome_produto, categorias_validas):
    prompt_sistema = \
    f"""
    Você é um categorizador de produtos.
    Quero que você escreva somente o nome da categoria.
    Você deve escolher uma categoria das lista abaixo:
    #### Lista de categorias válidas
    {categorias_validas} 
    O Formato de saída deve ser apenas nome da categoria, sem numero e ponto
    #### Exemplo:
    bola de tênis
    Esportes
    """
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": prompt_sistema
        },
        {
        "role": "user",
            "content": nome_produto
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    )

    for choice in response.choices:
        print(f"A categoria do produto informado {nome_produto} é {choice.message.content}")
        print("---------------------\n")
        
print("Digite as categorias válidas:")
categorias_validas = input()

while True:
    print("Digite o nome do produto:")
    nome_produto = input()
    categoriza_produto(nome_produto, categorias_validas)
