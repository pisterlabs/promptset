#print das frases no fim
import os
import pandas as pd
import Levenshtein
import openai
import time
import json

# Funções para a tradução
def translate_to_modern_portuguese(text, api_key):
    print(f"Texto original: {text}")
    print("Traduzindo...")
    openai.api_key = api_key
    prompt = [{
        "role": "user",
        "content": f"corrija a grafia da seguinte frase, corrigindo numerais por numeros em extenso e "
                   f"corrigindo a acentuação, retorne apenas a "
                   f"frase corrigida: {text}"
    }]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=prompt, temperature=0.2, max_tokens=200)
    texto_traduzido = response['choices'][0]['message']['content'].strip()
    print(f"Texto traduzido: {texto_traduzido}")
    time.sleep(1)
    return texto_traduzido

# Carrega o dataset
df = pd.read_excel('metadata_train_normal.xlsx', engine='openpyxl')

#pre processamento
df['n_text'] = df['text'].str.lower()
df['n_text'] = df['text'].str.strip()
df['text'] = df['text'].str.replace(r'^—', '', regex=True)

df['n_transcript_whisper'] = df['transcript_whisper'].str.lower()
df['n_transcript_whisper'] = df['transcript_whisper'].str.strip()

df['n_transcript_mms'] = df['transcript_mms'].str.lower()
df['n_transcript_mms'] = df['transcript_mms'].str.strip()

df.fillna('', inplace=True)

API_KEY = ''

# Lista para armazenar frases antes e depois da tradução
translated_texts = []

# calcula a similaridade
for index in range(len(df)):
    if pd.isna(df.at[index, 'text']) or df.at[index, "text"] == '':
        print(f"Skipping line {index + 1} due to empty 'text' field.")
        continue

    similarity_1_2 = Levenshtein.ratio(df.at[index, 'n_text'], df.at[index, 'n_transcript_mms'])
    similarity_1_3 = Levenshtein.ratio(df.at[index, 'n_text'], df.at[index, 'n_transcript_whisper'])

    avg_similarity = (similarity_1_2 + similarity_1_3) / 2
    print(f"Linha {index + 1}:")
    print(f"Frases: {df.at[index, 'text']}")
    print(f"Frases mms: {df.at[index, 'transcript_mms']}")
    print(f"Frases whisper: {df.at[index, 'transcript_whisper']}")
    print(f"Média de Similaridade: {avg_similarity:.4f}")

    if 0 < avg_similarity < 0.90:
        print("*********BAIXA SIMILARIDADE**************")
        original_text = df.at[index, 'text']
        translated_text = translate_to_modern_portuguese(df.at[index, 'text'], API_KEY)
        df.at[index, 'text'] = translated_text

        # Armazenar o texto original e o texto traduzido
        translated_texts.append({
            'original_text': original_text,
            'translated_text': translated_text
        })

# Salvar o DataFrame atualizado em um novo arquivo Excel
df.drop(['n_transcript_whisper', 'n_transcript_mms', 'n_text'], axis=1, inplace=True)
df.to_excel('updated_metadata_train_norm.xlsx', engine='openpyxl', index=False)
print("O arquivo Excel foi salvo com sucesso!")

# Imprimir a lista de textos traduzidos
print("\nTextos que passaram pela tradução:")
for i, text_pair in enumerate(translated_texts):
    print(f"{i+1}. Original: {text_pair['original_text']}")
    print(f"   Traduzido: {text_pair['translated_text']}")
