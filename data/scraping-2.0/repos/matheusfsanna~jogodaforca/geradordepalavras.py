import openai
import pandas as pd
#openai.api_key = "sua-chave"

# Função para obter o tema escolhido pelo usuário
def obter_tema_do_usuario():
    return input("Digite o tema desejado para gerar palavras: ")

# Obtém o tema escolhido pelo usuário
tema_escolhido = obter_tema_do_usuario()

# Cria a solicitação para o ChatGPT
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": f"Gere palavras, com índice, para um jogo da forca com o tema {tema_escolhido}"
        },
        {
            "role": "user",
            "content": f"Gere 10 palavras para um jogo da forca com o tema {tema_escolhido}"
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

# Processa a resposta para criar um dicionário com índice numérico
def extrair_palavras_com_indice(resposta):
    linhas = resposta['choices'][0]['message']['content'].split('\n')
    palavras_com_indice = [linha.split('. ')[1] for linha in linhas if linha.startswith(('\d', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))]
    dicionario_palavras = {indice: palavra.upper() for indice, palavra in enumerate(palavras_com_indice, 1)}
    return dicionario_palavras

# Usa a função para obter o dicionário de palavras com índice
palavras_com_indice = extrair_palavras_com_indice(response)

# Exibe o dicionário de palavras com índice
df = pd.DataFrame(list(palavras_com_indice.items()), columns=['Índice', 'Palavra'])

# Salva o DataFrame em um arquivo CSV
df.to_csv('palavras.csv', index=False)

print(f"Arquivo CSV 'palavras.csv' salvo com sucesso para o tema '{tema_escolhido}'.")
