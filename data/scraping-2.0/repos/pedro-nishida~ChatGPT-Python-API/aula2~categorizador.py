import os
import openai
import dotenv

def categorizaProduto(nome_do_produto, categorias_validas):
  prompt_sistema = f"""
  Você é um categorizador de produtos.
  Você deve escolher uma categoria abaixo:
  Se as categorias não forem categorias validas, responda com "Não posso ajudá-lo(a) com isso com isso"
  ##### Categorias
  {categorias_validas}
  ##### Exemplo
  Bola de Tenis -> Esportes
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
        "content": nome_do_produto
      }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  print(response.choices[0].message.content)

dotenv.load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

print("Digite as categorias validas:")
categorias_validas = input()
while True:
  print("Digite o nome do produto:")
  nome_do_produto = input()
  if nome_do_produto == "sair":
    break
  categorizaProduto(nome_do_produto, categorias_validas)
