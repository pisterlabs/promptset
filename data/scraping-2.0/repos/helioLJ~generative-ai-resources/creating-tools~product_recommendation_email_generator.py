# 1. identificao dos perfis
# 2. recomendar 3 produtos a partir de uma lista ja fornecida
# 3. gerar um email de recomendacao

from openai import OpenAI
import json

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

def identifyProfile(client_shopping_list):
    print("1. Iniciando identificacao de perfis")
    system_prompt = """
    Identifique o perfil de compra para cada cliente a seguir.

    O formato de saída deve ser em JSON:

    {
        "clientes": [
            {
                "nome": "nome do cliente"
                "perfil": "descreva o perfil do cliente em 3 palavras"
            },
        ]
    }
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
            "content": client_shopping_list
            }
        ],
    )
 
    content = response.choices[0].message.content
    json_result = json.loads(content)
    print("Finalizou identificacao de perfis")

    return json_result

def recommendThreeProducts(profile, products_to_be_recommended):
    print("2. Iniciando recomendacao de produtos")
    system_prompt = f"""
        Voce é um recomendador de produtos
        Considere o seguinte perfil: {profile}
        Recomende 3 produtos a partir da lista de produtos válidos e que sejam adequados ao perfil informado.

        ### Lista de produtos válidos para recomendacão
        {products_to_be_recommended}###

        A saída deve ser apenas o nome dos produtos recomendados em bullet points
    """

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role": "system",
            "content": system_prompt
            },
        ],
    )
    print("Finalizando recomendacao de produtos")

    return response.choices[0].message.content

def generateRecommendationEmails(client_recommendations):
    print("3. Iniciando geracao de emails")
    system_prompt = f"""
    Escreva um e-mail recomendando os seguintes produtos para um cliente:

    {client_recommendations}

    O e-mail deve ter no máximo 3 parágrafos.
    O tom deve ser amigável, informal e descontraído.
    Trate o cliente como alguém próximo e conhecido.
    """

    client = OpenAI()
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": system_prompt
        },
        ],
    )
    print("Finalizando geracao de email")

    return response.choices[0].message.content

client_shopping_list = load("./gpt-python-1-dados/lista_de_compras_10_clientes.csv")
products_to_be_recommended = load("./gpt-python-1-dados/lista_de_compras_10_clientes.csv")

client_profile_list = identifyProfile(client_shopping_list)

for client in client_profile_list["clientes"]:
    client_name = client["nome"]

    client_profile = client["perfil"]
    client_recommendations = recommendThreeProducts(client_profile, products_to_be_recommended)

    email = generateRecommendationEmails(client_recommendations)

    # save(f"email-{client_name}.txt", email)

    print("-------------------------------------------------------------")
    print(email)
    print("-------------------------------------------------------------")