import pandas as pd
import openai

# Chave de API da OpenAI
api_key = 'API KEY'

# Função para gerar um resumo com base no feedback usando a API GPT-3
def gerar_resumo(feedback):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {
          "role": "system",
          "content": "Você é um especialista em resumir feedbacks de clientes."
      },
      {
          "role": "user",
          "content": f"Resuma o feedback a seguir em até 50 caracteres: '{feedback}'. Caso não exista feedback, responda N/A."
      }
    ]
  )
    return response.choices[0].message.content.strip('\"')

# Carregando os dados
arquivo = 'feedback_form.csv'
dados = pd.read_csv(arquivo)

# Adicionando uma coluna de resumo gerado pelo GPT, com base no feedback
dados['resumo_feedback'] = dados['feedback'].apply(gerar_resumo)

# Padronizando a formatação da coluna de data
dados['data_envio'] = pd.to_datetime(dados['data_envio'], format='%d/%m/%Y')

# Verificando e tratando respostas em branco ou inconsistentes (preenchendo com "N/A")
dados = dados.fillna('N/A')

# Salvando os dados transformados em um novo arquivo CSV
saida = 'dados_processados.csv'
dados.to_csv(saida, index=False)