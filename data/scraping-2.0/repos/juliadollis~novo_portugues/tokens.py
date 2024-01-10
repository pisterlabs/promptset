import openai
import pandas as pd
import os
import tiktoken

openai.api_key = API KEY

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def batch_translate(texts, max_tokens=1000):
    start = 0
    end = 0
    translated_texts = []
    while end < len(texts):
        tokens = 0
        batch = []

        while end < len(texts):
            current_text_token_count = num_tokens_from_string(texts[end])
            print(f"Tokens no texto {end}: {current_text_token_count}")  # Print para conferência

            if tokens + current_text_token_count <= max_tokens:
                tokens += current_text_token_count
                batch.append(texts[end])
                end += 1
            else:
                break

        print(f"Traduzindo o lote de tokens de índice {start} a {end - 1}...")

        batch_str = " || ".join(batch)
        print(f"Frases antes da tradução: {batch_str}")

        prompt = [{"role": "user",
                   "content": f"corrija a grafia das seguintes frases do portugues antigo para o portugues atual, "
                              f"corrigindo numerais e numeros por numeros em extenso, corrigindo a acentuação e "
                              f"abreviações,  retorne apenas as frases corrigidas separadas por ||, sem adições:" +
                              batch_str}]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            temperature=0.0,
            max_tokens=max_tokens
        )

        translations = response['choices'][0]['message']['content'].split("||")
        translated_batch = [t.strip() for t in translations]

        print(f"Frases após a tradução: {translated_batch}")

        translated_texts.extend(translated_batch)
        print(f"Traduções concluídas para o lote de índice {start} a {end - 1}")

        start = end

    return translated_texts

# Leitura do arquivo Excel
df = pd.read_excel("devtok.xlsx", engine='openpyxl', sheet_name='Planilha1')

if df.empty:
    print("O DataFrame está vazio. Nada para traduzir.")
else:
    try:
        transcripts = df['transcript'].tolist()
        translated_texts = batch_translate(transcripts)
        df['transcript'] = pd.Series(translated_texts)
    except Exception as e:
        print("Erro encontrado:", e)

    df.to_excel("dev_tok_att.xlsx", engine='openpyxl', index=False)
    print("O arquivo Excel foi salvo com sucesso!")
