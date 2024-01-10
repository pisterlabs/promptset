import os
import openai
import dotenv
import json

def carrega(nome_do_arquivo):
    try:
        with open(nome_do_arquivo, "r") as arquivo:
            dados = arquivo.read()
            return dados
    except IOError as e:
        print(f"Erro no carregamento de arquivo: {e}")

def salva(nome_do_arquivo, conteudo):
    try:
        with open(nome_do_arquivo, "w", encoding="utf-8") as arquivo:
            arquivo.write(conteudo)
    except IOError as e:
        print(f"Erro ao salvar arquivo: {e}")

def identifica_perfis(lista_de_compras_por_cliente):
  print("1. Iniciando identificação de perfis")
  prompt_sistema = """
  Identifique o perfil de compra para cada cliente a seguir.

  O formato de saída deve ser rm JSON:

  {
    "clientes": [
      {
        "nome": "nome do cliente",
        "perfil": "descreva perfil do cliente em 3 palavras"
      }
    ]
  }
  """

  resposta = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
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

  conteudo = resposta.choices[0].message.content
  json_resultado = json.loads(conteudo)
  print("1. Finalzando os Perfis identificados")
  return json_resultado
  
def recomenda_produtos(perfil, lista_de_produtos):
  print("2. Iniciando recomendação de produtos")
  prompt_sistema = """
  Você é um recomendador de produtos.
  Considere o seguinte perfil: {perfil}
  Recomende 3 produtos a partior da lista de produtos validos e que sejam adequados ao perfil informado.
  
  #### Lista de produtos válidos para recomendação
  {lista_de_produtos}

  A saída deve ser apenas o nome dos produtos recomendados em bullet points.
  """

  resposta = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {
        "role": "system",
        "content": prompt_sistema
      }
    ]
  )

  conteudo = resposta.choices[0].message.content
  print("2. Finalizando recomendação de produtos")
  return conteudo

def escreve_email(recomendacoes):
  print("3. Iniciando escrita de email")
  prompt_sistema = f"""
  Escreva um e-mail recomendando os seguintes produtos para um cliente:

  {recomendacoes}

  O e-mail deve ter no máximo 3 parágrafos.
  O tom deve ser amigável, informal e descontraído.
  Trate o cliente como alguém próximo e conhecido.
  """


  resposta = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {
        "role": "system",
        "content": prompt_sistema
      }
    ]
  )

  conteudo = resposta.choices[0].message.content
  print("3. Finalizando escrita de email")
  return conteudo

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

lista_de_produtos = carrega("aula6\dados\lista_de_produtos.txt")
lista_de_compras_por_cliente = carrega("aula6\dados\lista_de_compras_10_clientes.csv")
perfis = identifica_perfis(lista_de_compras_por_cliente)
for cliente in perfis["clientes"]:
  nome_do_cliente = cliente["nome"]
  print(f"Iniciando recomendações para o cliente: {nome_do_cliente}")
  recomendacoes = recomenda_produtos(cliente["perfil"],lista_de_produtos)
  email = escreve_email(recomendacoes)
  salva(f"email-{nome_do_cliente}.txt", email)
