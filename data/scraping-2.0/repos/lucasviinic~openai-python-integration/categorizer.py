from openai import OpenAI
import dotenv
import os


def product_categorizer(product_name: str, valid_categories: str) -> str:
    prompt_system = f"""
    Você é um categorizador de produtos.
    Você deve escolher uma categoria da lista abaixo:
    Se as categorias informadas não forem categorias válidas, responda com "Não posso ajudá-lo com isso."
    ###### Lista de categorias válidas
    {valid_categories}
    ###### Exemplo
    bola de tênis
    Esportes
    """

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": prompt_system
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
    presence_penalty=0
    )

    category = response.choices[0].message.content
    return category

dotenv.load_dotenv()

client = OpenAI()
client.api_key = os.getenv('OPENAI_API_KEY')

valid_categories = input("Digite as categorias válidas: ")
while True:
    product_name = input("Digite o nome do produto: ")
    print(product_categorizer(product_name, valid_categories))