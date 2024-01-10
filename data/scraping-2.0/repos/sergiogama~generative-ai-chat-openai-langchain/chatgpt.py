# Autor: Sergio Gama
# Data: Julho/2023
# Importa as bibliotecas necessárias
import os
import pdfplumber
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

input_file = 'data.txt'  # Define o arquivo de entrada
# Verifica se os dados de entrada estão no formato PDF ou formato de texto simples
pdf = 0  # Defina pdf para 1 se os dados estiverem no formato PDF, caso contrário, defina como 0 (assumindo que é texto simples)
if pdf == 1:
    # Se os dados estiverem no formato PDF, extrai o texto do PDF e salva em um arquivo local
    pdf_file = 'Reforma-Tributaria-2023.pdf'
    input_file = 'data_from_pdf.txt'
    with pdfplumber.open(pdf_file) as pdf:
        # Remove o arquivo local se já existir
        if os.path.exists(input_file):
            os.remove(input_file)

        # Lê todas as páginas do PDF e salva em um arquivo local
        for i in range(0, len(pdf.pages)):
            page = pdf.pages[i]
            text = page.extract_text()

            with open(input_file, 'a') as f:
                f.write(text)
                f.write('\n')
                f.close()
                
loader = TextLoader(input_file, autodetect_encoding=True)  # Cria um carregador de texto para os dados de texto simples

# Cria um índice usando o VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])

# Inicia um loop para consultar interativamente o índice
while True:
    query = input("Digite a pergunta: ")  # Solicita ao usuário que digite uma pergunta
    print(index.query(query))  # Imprime o resultado da consulta

# O código abaixo (comentado) mostra algumas consultas de exemplo que podem ser usadas para recuperar informações do índice:
# query = "O que seria o IVA?"  # Pergunta para os dados do "Reforma-Tributaria-2023.pdf"
# print(index.query(query))

# query = "Quais produtos têm, e qual é o mais barato?"  # Pergunta para os dados do "data.txt"
# print(index.query_with_sources(query))

# query = "Quanto custa o sapato, e quantos têm em estoque?"  # Pergunta para os dados do "data.txt"
# print(index.query_with_sources(query))
