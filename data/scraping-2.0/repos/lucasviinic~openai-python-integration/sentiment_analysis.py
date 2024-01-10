import os
import openai
from openai import OpenAI
import dotenv
import time


def sentiment_analysis(product_name: str):
    system_prompt = """
    Você é um analisador de sentimentos de avaliações de produtos.
    Escreva um parágrafo com até 50 palavras resumindo as avaliações e depois atribua qual o sentimento geral para o produto.
    Identifique também 3 pontos fortes e 3 pontos fracos identificados a partir das avaliações.

    #### Formato de saída

    Nome do produto: 
    Resumo das avaliações:
    Sentimento geral: [aqui deve ser POSITIVO, NEUTRO ou NEGATIVO]
    Pontos fortes: [3 bullet points]
    Pontos fracos: [3 bullet points]
    """

    user_prompt = load(f"./data/avaliacoes-{product_name}.txt")
    print(f"Iniciando a análise do produto: {product_name}")

    attempts = 0
    wait_time = 5
    while attempts < 3:
        attempts += 1
        print(f"Tentativa {attempts}")
        try:
            resposta = client.chat.completions.create(
                model = "gpt-3.5-turbo",
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )

            save(f"./data/analise-{product_name}", resposta.choices[0].message.content)
            print("Análise concluída com sucesso.")
            return

        except openai.RateLimitError as e:
            print(f"Erro de limite de taxa: {e}")
            time.sleep(wait_time)
            wait_time *= 2 # exponential recoil
        except openai.AuthenticationError as e:
            print(f"Erro de autenticacao: {e}")
        except openai.APIError as e:
            print(f"Erro de API: {e}")
            time.sleep(5)
        

def load(file_name: str) -> str:
    try:
        with open(file_name, "r") as file:
            data = file.read()
            return data
    except IOError as e:
        print(f"Erro no carregamento de arquivo: {e}")

def save(file_name: str, content: str):
    try:
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(content)
    except IOError as e:
        print(f"Erro ao salvar arquivo: {e}")

dotenv.load_dotenv()

client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

product_list = ["DVD player automotivo", "Esteira elétrica para fitness", "Grill elétrico para churrasco", "Mixer de sucos e vitaminas", "Tapete de yoga", "Miniatura de carro colecionável", "Balança de cozinha digital", "Jogo de copos e taças de cristal", "Tabuleiro de xadrez de madeira", "Boia inflável para piscina"]
for product_name in product_list:
    sentiment_analysis(product_name)