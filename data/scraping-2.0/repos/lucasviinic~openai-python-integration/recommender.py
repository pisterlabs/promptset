import os
import openai
import dotenv
import json
import time


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

def profile_identifier(purchase_list_per_clients: list) -> dict:
    print("1. Iniciando identificação de perfis")
    system_prompt = """
    Identifique o perfil de compra para cada cliente a seguir.

    O formato de saída deve ser em JSON:

    {
        "clientes": [
            {
                "nome": "Nome do cliente",
                "perfil": "descreva o perfil do cliente em 3 palavras"
            }
        ]
    }
    """
    attempts = 0
    wait_time = 5
    while attempts < 3:
        attempts += 1
        print(f"Tentativa {attempts}")
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": purchase_list_per_clients
                    }
                ]
            )

            content = response.choices[0].message.content
            json_result = json.loads(content)
            print("1. Finalizou identificação de perfis")
            return json_result

        except openai.RateLimitError as e:
            print(f"Erro de limite de taxa: {e}")
            time.sleep(wait_time)
            wait_time *= 2
        except openai.AuthenticationError as e:
            print(f"Erro de autenticacao: {e}")
        except openai.APIError as e:
            print(f"Erro de API: {e}")
            time.sleep(5)
    
def recommendes_product(profile, product_list):
    print("2. Iniciando recomendação de produtos")
    system_prompt = f"""
    Você é um recomendador de produtos.
    Considere o seguinte perfil: {profile}
    Recomende 3 produtos a partir da lista de produtos válidos e que sejam adequados ao perfil informado.
    
    #### Lista de produtos válidos para recomendação
    {product_list}
    
    A saída deve ser apenas o nome dos produtos recomendados em bullet points
    """

    attempts = 0
    wait_time = 5
    while attempts < 3:
        attempts += 1
        print(f"Tentativa {attempts}")
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ]
            )

            content = response.choices[0].message.content
            print("2. Finalizando recomendação de produtos")
            return content
    
        except openai.RateLimitError as e:
            print(f"Erro de limite de taxa: {e}")
            time.sleep(wait_time)
            wait_time *= 2
        except openai.AuthenticationError as e:
            print(f"Erro de autenticacao: {e}")
        except openai.APIError as e:
            print(f"Erro de API: {e}")
            time.sleep(5)

def write_email(recomendations):
    print("3. Escrevendo e-mail de recomendação")
    system_prompt = f"""
    Escreva um e-mail recomendando os seguintes produtos para um cliente:

    {recomendations}

    O e-mail deve ter no máximo 3 parágrafos.
    O tom deve ser amigável, informal e descontraído.
    Trate o cliente como alguém próximo e conhecido.
    """

    attempts = 0
    wait_time = 5
    while attempts < 3:
        attempts += 1
        print(f"Tentativa {attempts}")
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ]
            )

            content = response.choices[0].message.content
            print("3. Finalizando escrita do e-mail")
            return content
    
        except openai.RateLimitError as e:
            print(f"Erro de limite de taxa: {e}")
            time.sleep(wait_time)
            wait_time *= 2
        except openai.AuthenticationError as e:
            print(f"Erro de autenticacao: {e}")
        except openai.APIError as e:
            print(f"Erro de API: {e}")
            time.sleep(5)

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

product_list = load("./data/lista_de_produtos.txt")
purchase_list_per_clients = load("./data/lista_de_compras_10_clientes.csv")
profiles = profile_identifier(purchase_list_per_clients)

for client in profiles["clientes"]:
    client_name = client["nome"]
    print(f"Iniciando recomendação para o cliente {client_name}")
    
    recomendations = recommendes_product(client["perfil"], product_list)
    email = write_email(recomendations)
    save(f"email-{client_name}.txt", email)
