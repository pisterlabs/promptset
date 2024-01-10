import os
import openai
import dotenv
import json
import time

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
    print('1. indentificcando perfil')
    prompt_sistema = """
Identifique o perfil de compra para cada cliente a seguir.

O formato de saída deve ser em JSON:
ex: 
{
    "clientes": [
        {
            "nome": "nome do cliente",
            "perfil": "descreva o perfil do cliente em 3 palavras"
        }
    ]
}


  """
    tentativas = 0
    tempo_exponencial = 5
    while tentativas < 3:
        try:
            tentativas += 1
            print(f'tentativa {tentativas}')
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
            print('1. OK, perfil identifcado')
            return json_resultado
        except openai.error.AuthenticationError as e:
            print(f'ERRO DE AUTENTICAÇÃO: {e}')
        except openai.error.APIError as e:
            print(f'ERRO DE API: {e}')
            time.sleep(5)
        except openai.error.RateLimitError as e:
            print(f'ERRO LIMITE DE TAXA: {e}')
            time.sleep(tempo_exponencial)
            tempo_exponencial *= 2



def recomenda_produto(perfil,lista_de_produtos):
    print('2. recomendando produto')
    prompt_sistema = f"""
    Você é um recomendador de produtos.
    Considere o seguinte perfil: {perfil}
    Recomende 3 produtos a partir da lista de produtos válidos e que sejam adequados ao perfil informado.
  
    #### Lista de produtos válidos para recomendação
    {lista_de_produtos}

    A saída deve ser apenas o nome dos produtos recomendados em bullet points. 
    """
    tentativas = 0
    tempo_exponencial = 5
    while tentativas < 3:
        try:
            tentativas += 1
            print(f'tentativa {tentativas}')
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
            return conteudo
        except openai.error.AuthenticationError as e:
            print(f'ERRO DE AUTENTICAÇÃO: {e}')
        except openai.error.APIError as e:
            print(f'ERRO DE API: {e}')
            time.sleep(5)
        except openai.error.RateLimitError as e:
            print(f'ERRO LIMITE DE TAXA: {e}')
            time.sleep(tempo_exponencial)
            tempo_exponencial *= 2


def escreve_email(recomendacoes):
    print('3. escrevendo email')
    prompt_sistema = f"""
        Gere um emial com no maximo 3 paragrafos apresentando esses produtos: {recomendacoes}
        o tom deve deve ser amigavel e descontraido e levemente informal
        """
    tentativas = 0
    tempo_exponencial = 5
    while tentativas < 3:
        try:
            tentativas+=1
            print(f'tentativa {tentativas}')
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
            print('3. OK, email enviado!')
            return conteudo
        except openai.error.AuthenticationError as e:
            print(f'ERRO DE AUTENTICAÇÃO: {e}')
        except openai.error.APIError as e:
            print(f'ERRO DE API: {e}')
            time.sleep(5)
        except openai.error.RateLimitError as e:
            print(f'ERRO LIMITE DE TAXA: {e}')
            time.sleep(tempo_exponencial)
            tempo_exponencial *= 2


dotenv.load_dotenv()
openai.api_key = os.getenv("API_GPT")
lista_de_produtos = carrega("./dados/lista_de_produtos.txt")
lista_de_compras_por_cliente = carrega("./dados/clientes_10.csv")
perfis = identifica_perfis(lista_de_compras_por_cliente)
for cliente in perfis['clientes']:
    print(cliente["nome"])
    recomendacoes = recomenda_produto(cliente['perfil'],lista_de_produtos)
    print(recomendacoes)
    conteudo = escreve_email(recomendacoes)
    salva(f'./dados/email_{cliente["nome"]}.txt',conteudo)

