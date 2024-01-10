import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
import time


# Funções para a tradução
def translate_to_modern_portuguese(text, api_key):
    print(text)
    print("Traduzindo...")
    openai.api_key = api_key
    prompt = [{
        "role": "user",
        "content": f"corrija a grafia da seguinte frase do portugues antigo para o portugues atual, corrigindo "
                   f"numerais e numeros por numeros em extenso e corrigindo a acentuação, retorne apenas a "
                   f"frase corrigida: {text}"
    }]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=prompt, temperature=0.2, max_tokens=200)
    texto_traduzido = response['choices'][0]['message']['content'].strip()
    print(texto_traduzido)
    time.sleep(21)
    return texto_traduzido


# Carregar e processar dados
vectorizer = TfidfVectorizer()

# Carregar o dataset
df = pd.read_excel('metadata_train.xlsx', engine='openpyxl')
df.fillna('', inplace=True)

all_sentences = pd.concat([df['text'], df['transcript_mms'], df['transcript_whisper']])
vectorizer.fit(all_sentences)

tfidf_coluna1 = vectorizer.transform(df['text'])
tfidf_coluna2 = vectorizer.transform(df['transcript_mms'])
cosine_sim_1_2 = cosine_similarity(tfidf_coluna1, tfidf_coluna2)

tfidf_coluna3 = vectorizer.transform(df['transcript_whisper'])
cosine_sim_1_3 = cosine_similarity(tfidf_coluna1, tfidf_coluna3)

API_KEY = 'KEY'

for index in range(len(df)):
    avg_similarity = (cosine_sim_1_2[index][index] + cosine_sim_1_3[index][index]) / 2
    print(f"Linha {index + 1}:")
    print(f"  Média de Similaridade: {avg_similarity:.4f}")

    if 0 < avg_similarity < 0.85:
        df.at[index, 'text'] = translate_to_modern_portuguese(df.at[index, 'text'], API_KEY)

# Salvar o DataFrame atualizado em um novo arquivo Excel
df.to_excel('updated_metadata_train.xlsx', engine='openpyxl', index=False)
print("O arquivo Excel foi salvo com sucesso!")
