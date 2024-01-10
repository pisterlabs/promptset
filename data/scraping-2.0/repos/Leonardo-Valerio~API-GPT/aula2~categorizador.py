import os
import openai
import dotenv

def categorizador(nome_produto, categorias):
    prompt = f"""
        Você é um categorizador de produtos.
        e deve categorizar os produtos de acordo com a lista de categorias válidas
        ###
        Lista de categorias válidas
        ###
        {categorias}
        ###
        a sáida da resposta deve ser: Exemplo
        bola de futebol
        Esportes
        ###
        se a categoria for uma categoria invalida para um produto qualquer retorne que essa categoria é invalida
    
    """
    resposta = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {
          "role": "system",
          "content": prompt
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
      presence_penalty=0
    )

    print(resposta.choices[0].message.content)


dotenv.load_dotenv()
openai.api_key = os.getenv("API_GPT")
categoria = input('qual categoria deseja adicionar? ')
qtd = int(input('quer categorizar quantos produtos? '))
for i in range(qtd):
    nome = input('digite o nome do produto para categorizar: ')

    categorizador(nome,categoria)