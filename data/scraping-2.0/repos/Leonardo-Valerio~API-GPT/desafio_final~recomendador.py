import openai
import os
import dotenv

def ler_arquivo(nome_arquivo):
    try:
        with open(nome_arquivo, 'r') as arquivo:
            dados = arquivo.readlines()
            return dados
    except:
        print('erro ao ler arquivo')

def gera_arquivo(nome_arquivo,conteudo):
    try:
        with open(nome_arquivo, 'w',encoding='utf-8') as arquivo:
            dados = arquivo.write(conteudo)
    except:
        print('erro ao gerar arquivo')

dotenv.load_dotenv()
openai.api_key = os.getenv('API_GPT')
prompt_systema = """
    Você é um gerador de email, e vai analisar im arquivo que conterá oque cada cliente comprou e me gererá um email
    recomemndando 3 produtos para cada cliente de acordo com as últimas compras dele, ou seja , analise, os gostos dele
    e me recomende 3 produtos de acordo com seus gostos
    ### Formato de sáida:
    paragrafo apresentativo do e-commerce chamado 'compre já', com até 50 palavras
    exemplo: Olá (nome do cliente), ....
    
    produto 1: apresentando o produto com até 50 palavras no maximo
    produto 2: apresentando o produto com até 50 palavras no maximo
    produto 3: apresentando o produto com até 50 palavras no maximo
"""
prompt_usuario = ler_arquivo('./dados/cliente_10.txt')
for i in range(10):
    resposta = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = [
            {
                'role': 'system',
                'content': prompt_systema
            },
            {
                'role': 'user',
                'content': prompt_usuario[i]
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0

    )

    gera_arquivo(f"./dados/cliente{i}_10.txt",resposta.choices[0].message.content)
    print(f'email {i}')