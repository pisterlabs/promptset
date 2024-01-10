import time
import openai
from openai import OpenAI
client = OpenAI()

def load(file_name):
    try:
        with open(file_name, "r") as file:
            data = file.read()
            return data
    except IOError as e:
        print(f"Error: {e}")

def save(file_name, content):
    try:
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(content)
    except IOError as error:
        print(f"Error: {error}")

def analyzeFeelings(product_name):
    system_prompt = f"""
    Voce é um analisador de sentimentos de avaliacões de produtos.
    Escreva um parágrafo com até 50 palavras resumindo as avaliacões e depois atribual qual o sentimento geral para o produto.
    Identifique também 3 pontos fortes e 3 pontos fracos identificados a partir das avaliacões.

    ### Formato de saída

    Nome do produto:
    Resumo das avaliacões:
    Sentimento geral: [aqui deve ser POSITIVO, NEUTRO ou NEGATIVO]
    Pontos fortes: [3 bullet points]
    Pontos fracos: [3 bullet points]
    ###
    """

    user_prompt = load(f"./gpt-python-1-dados/avaliacoes-{product_name}.txt")
    tries = 0
    wait_time = 0
    while tries < 3:
        tries += 1
        print(f"Tentativas: {tries}")
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                    "role": "system",
                    "content": system_prompt
                    },
                    {
                    "role": "user",
                    "content": user_prompt
                    }
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return
        except openai.AuthenticationError as error:
            print(f"Authentication Error: {error}")
        except openai.APIError as error:
            print(f"API Error: {error}")
        except openai.RateLimitError as error:
            print(f"Rate Limit Error: {error}")
            time.sleep(wait_time) # recuo exponencial
            wait_time *= 2

    print("Análise feita com sucesso")
    save(f"./gpt-python-1-dados/analise-{product_name}", response.choices[0].message.content)

products_list = ["Tapete de yoga", "Tabuleiro de xadrez de madeira"]

for product_name in products_list:
    analyzeFeelings(product_name)