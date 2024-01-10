import os
import openai
import dotenv
import time

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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

def analise_sentimento(nome_do_produto):
    prompt_sistema = """
    Você é um analisador de sentimentos de avaliações de produtos.
    Escreva um parágrafo com até 50 palavras resumindo as avaliações e depois atribua qual o sentimento geral para o produto.
    Identifique também 3 pontos fortes e 3 pontos fracos identificados a partir das avaliações.

    #### Formato de saída

    Nome do produto: 
    Resumo das avaliações:
    Sentimento geral: 
    Pontos fortes: 
    Pontos fracos:
    """

    print(f"Iniciando a análise do produto: {nome_do_produto}")
    prompt_usuario = carrega(f"aula5\\analise_sentimento\\dados\\avaliacoes-{nome_do_produto}.txt")
    tempo_de_espera = 5
    tentativas = 0
    while tentativas < 3:
        tentativas += 1
        print(f"Tentativa {tentativas} de 3")
        try:
            resposta = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = [
                    {
                        "role": "system",
                        "content": prompt_sistema
                    },
                    {
                        "role": "user",
                        "content": prompt_usuario
                    }
                ]
            )
            salva(f"aula5\\analise_sentimento\\dados\\avaliacoes-{nome_do_produto}", resposta.choices[0].message.content)
            print("Análise concluída com sucesso!")
            return

        except openai.error.AuthenticationError as e:
            print(f"Erro na autenticação: {e}")
        except openai.error.APIError as e:
            print(f"Erro na API: {e}")
            time.sleep(5)
        except openai.error.rateLimiteError as e:
            print(f"Erro de autenticação: {e}")
            time.sleep(tempo_de_espera)
            tempo_de_espera *= 2



lista_de_produtos = ["Tapete de yoga", "Tabuleiro de xadrez de madeira"] 
for nome_do_produto in lista_de_produtos:
    analise_sentimento(nome_do_produto)