# Importação de bibliotecas externas
import os
import openai
import dotenv
import time

# Definição de função que carrega o conteúdo do arquivo desejado na variável dados, com lançamento de exceção 
def carrega(nome_arquivo):
    try:
        with open(nome_arquivo, "r") as arquivo:
            dados = arquivo.read()
            return dados
    except IOError as e:
        print(f"Erro no carregamento de arquivo: {e}")

# Definição de função que salva um conteúdo em um arquivo de texto, com lançamento de exceção
def salva(nome_arquivo, conteudo):
    try:
        with open(nome_arquivo, "w", encoding="utf-8") as arquivo:
            arquivo.write(conteudo)
    except IOError as e:
        print(f"Erro ao salvar arquivo: {e}")

# Definição de funçao que realiza a análise de sentimento de um produto
def analise_sentimento(nome_produto):
    # Mensagem ao usuário para indicar o início de cada análise de produto
    print(f"Realizando a análise do produto {nome_produto}")

    # Declaração de variável que representará o prompt do sistema e atribuição de prompt
    prompt_sistema = """
    Você é um analisador de sentimentos de avaliações de produtos.
    Escreva um parágrafo com até 50 palavras resumindo as avaliações e depois atribua qual o sentimento geral para o produto.
    Identifique também 3 pontos fortes e 3 pontos fracos identificados a partir das avaliações.

    #### Formato de saída

    Nome do produto: 
    Resumo das avaliações:
    Sentimento geral: [AQUI DEVE SER POSITIVO, NEGATIVO E NEUTRO]
    Pontos fortes: [3 BULLET POINTS]
    Pontos fracos: [3 BULLET POINTS]
    """

    # Declaração de variável que representará o prompt do usuário e atribuição do conteúdo de 
    # retorno da função de carregamento de arquivo
    prompt_usuario = carrega(f".venv/gpt-python-1-dados/avaliacoes-{nome_produto}.txt")
    
    # Declaração de variável e atribuição de valor que será o tempo de espera para as exceções 
    # baseadas em tempo
    tempo_espera = 5

    # Loop for que fará três chamadas em caso de exceção.
    for i in range(0,3):
        # Bloco a ser executado em caso para verificação de exceção
        try:
            # Construção de um chat com modelo fixo, sem parâmetros
            response = openai.ChatCompletion.create(
                # Declaração de variável e atribuição de valor do modelo escolhido
                model = "gpt-3.5-turbo",
                # Declaração de variável tipo lista de dicionários com os papéis e prompts
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
            # Chamada de função que cria um arquivo com os dados de retorno do chat
            salva(f".venv/Scripts/main/analise-{nome_produto}.txt", response.choices[0].message.content)

            # Exibição de mensagem de conclusão da análise
            print(" -> Análise concluída!")

            # Keyword que paraliza o loop. Será atingida apenas se não houver exceção
            break
        # Tratamento da exceção AuthenticationError
        except openai.error.AuthenticationError as e:
            print(f"Erro de atuenticação: {e}")
        # Tratamento da excessão APIError com uso de pausa de 5 segundos a fim da API voltar dentro deste período
        except openai.error.APIError as e:
            print(f"Erro de API: {e}")
            time.sleep(tempo_espera)
        # Tratamento de excessão RateLimitError com uso de uma pausa exponencial para que o tempo de rate seja atualizado
        except openai.error.RateLimitError as e:
            print(f"Erro de limite de taxa: {e}")
            time.sleep(tempo_espera)
            tempo_espera *= 2


# Carretamento das variáveis de ambiente
dotenv.load_dotenv()
# Atribuiçã do valor da variável de ambiente OPENAI_API_KEY ao atributo api_key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Declaração da lista de produtos
lista_produtos = [
                  "Balança de cozinha digital",
                  "Boia inflável para piscina",
                  "DVD player automotivo",
                  "Esteira elétrica para fitness",
                  "Grill Elétrico para churrasco",
                  "Jogo de copos e taças de cristal",
                  "Miniatura de carro colecionável",
                  "Mixer de sucos e vitaminas",
                  "Tapete de yoga", 
                  "Tabuleiro de xadrez de madeira"
                 ]

# Loop for que iterará a lista de produtos e chamará a função analise_sentimento para cada iteração
for produto in lista_produtos:
    # Chamada da função que realiza a análise de sentimento
    analise_sentimento(produto)
