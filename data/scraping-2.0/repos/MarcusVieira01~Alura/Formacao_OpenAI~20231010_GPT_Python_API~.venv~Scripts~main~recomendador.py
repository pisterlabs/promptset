# Importação de bibliotecas externas
import os
import openai
import dotenv
import json

# Definição de função que carrega os dados de um arquivo em uma variável, com tratamento de exceção
def carrega(nome_do_arquivo):
    try:
        with open(nome_do_arquivo, "r") as arquivo:
            dados = arquivo.read()
            return dados
    except IOError as e:
        print(f"Erro no carregamento de arquivo: {e}")

# Definição de função que salva um conteúdo em um arquivo, com tratamento de exceção
def salva(nome_do_arquivo, conteudo):
    try:
        with open(nome_do_arquivo, "w", encoding="utf-8") as arquivo:
            arquivo.write(conteudo)
    except IOError as e:
        print(f"Erro ao salvar arquivo: {e}")

# Definição de função que identifica perfis de clientes construindo um chat e remetendo uma request 
# com os prompts de sistema e usuário
def identifica_perfis(lista_de_compras_por_cliente):
    # Exibição de mensagem para organização do output
    print("Identificando Perfis!")
    prompt_sistema = """
    Identifique o perfil de compra para cada cliente a seguir.

    O formato de saída deve ser em JSON:

    {
        "clientes":"[
            {
                "nome":"nome do cliente",
                "perfil":"descreva o perfil do cliente em 3 palavras"
            }
        ]
    }
    """
    # Construção de um chat com os papéis e prompts de sistema e usuário, atribuindo a mensagem de 
    # resposta à variável
    resposta = openai.ChatCompletion.create(
        # Declaração de variável com atribuição do modelo de GPT escolhido
        model="gpt-3.5-turbo",
        # Declaração de variável e atribuição de uma lista de dicionários com os prompts e papeis 
        # de sistema e usuário
        messages=[
            {
                "role": "system",
                "content": prompt_sistema
            },
            {
                "role": "user",
                "content": lista_de_compras_por_cliente
            }
        ]
    )
    # Exibição de mensagem para organização do output
    print("Encerrada identificação!")

    # Retorno o objeto JSON 
    return json.loads(resposta.choices[0].message.content)

# Definição de função que recomendará 3 produtos baseados no pefil de compra
def recomenda_produtos(perfil, lista_produtos):
    #Exibição de mensagem para organização do output
    print("Recomendando produtos")

    prompt_sistema = f"""
        Você é um recomendador de produtos.
        Considere o seguinte perfil: {perfil}
        Recomende 3 produtos a partir da lista de produtos válidos e que sejam adequados ao perfil informado.
        
        #### Lista de produtos válidos para recomendação
        {lista_produtos}
    
        A saída deve ser apenas o nome dos produtos recomendados em bullet points
    """
    # Construção de um chat com os papéis e prompts de sistema e usuário, atribuindo a mensagem de 
    # resposta à variável
    resposta = openai.ChatCompletion.create(
        # Declaração de variável com atribuição do modelo de GPT escolhido
        model="gpt-3.5-turbo",
        # Declaração de variável e atribuição de uma lista de dicionários com os prompts e papeis 
        # de sistema e usuário
        messages=[
            {
                "role": "system",
                "content": prompt_sistema
            }
        ]
    )
    # Exibição de mensagem para organização do output
    print("Encerrada recomendação!")

    # Retorno do conteúdo atribuído à variável resposta
    return resposta.choices[0].message.content

# Definição de função que retornará um conlteúdo de e-mail
def escreve_email(recomendacoes):
    #Exibição de mensagem para organização do output
    print("Escrevendo e-mail!")

    prompt_sistema = f"""
        Escreva um e-mail recomendando os seguintes produtos para um cliente:

        {recomendacoes}

        O e-mail deve ter no máximo 3 parágrafos.
        O tom deve ser amigável, informal e descontraído.
        Trate o cliente como alguém próximo e conhecido.
        Finalize escrevendo: Obrigado pela atenção. MARCUS VIEIRA
        """
    # Construção de um chat com os papéis e prompts de sistema e usuário, atribuindo a mensagem de 
    # resposta à variável
    resposta = openai.ChatCompletion.create(
        # Declaração de variável com atribuição do modelo de GPT escolhido
        model="gpt-3.5-turbo",
        # Declaração de variável e atribuição de uma lista de dicionários com os prompts e papeis 
        # de sistema e usuário
        messages=[
            {
                "role": "system",
                "content": prompt_sistema
            }
        ]
    )
    # Exibição de mensagem para organização do output
    print("Encerrada escrita de e-mail!")

    # Retorno do conteúdo atribuído à variável resposta
    return resposta.choices[0].message.content

# Carregamento das variáveis de ambiente
dotenv.load_dotenv()
# Atribuição do valor da variável de ambiente OPENAI_API_KEY ao atributo api_key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Declaração de variável e atribuição dos dados de retorno da função carrega()
lista_de_compras_por_cliente = carrega(".venv/gpt-python-1-dados/lista_de_compras_10_clientes.csv")

# Declaração de variável e atribuição dos dados de retorno da função carrega()
lista_produtos = carrega(".venv/gpt-python-1-dados/lista_de_produtos.txt")

# Declaração de variável e atribuição do valor de retorno da função identifica_perfis
perfis = identifica_perfis(lista_de_compras_por_cliente)

# Loop for que iterará o JSON retornado e atribuído à variável perfis, far á a lista de 
# produtos recomendados e escreverá um conteúdo de e-mail
for cliente in perfis["clientes"]:
    # Exibição de mensagem
    print(f'\nRecomendação para o cliente {cliente["nome"]}')
    # Chamada da função recomenda_produtos e atribuição do retorno em variável
    recomendacoes = recomenda_produtos(cliente["perfil"], lista_produtos)
    # Chamada da função escreve_email e atribuição do retorno em variável
    email = escreve_email(recomendacoes)
    # Chamada da função salva para salvar o conteúdo da variável email em um arquivo
    salva(f'.venv/Scripts/main/email-{cliente["nome"]}.txt', email)
    # Exibição do valor da variável e-mail
    print(email)
