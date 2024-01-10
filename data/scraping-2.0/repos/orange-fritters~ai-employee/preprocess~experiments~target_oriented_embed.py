from typing import List
import numpy as np
import openai
import json
import pandas as pd
import time


with open('server/model/utils/config.json') as f:
    config = json.load(f)
openai.api_key = config['chatgpt']['secret']


def get_embedding(texts: List[str]):
    result = openai.Embedding.create(
        engine="text-embedding-ada-002",
        input=texts)
    return result["data"][0]["embedding"]


def get_translation_target(text):
    prompt_message = [
        {"role": "assistant",
            "content": """
                - You must translate the following sentence into English.
                - Output only translation, No prose or explanation, indicators.
                """},
        {"role": "user",
            "content": f"Translate {text} to english"
         },
        {"role": "system",
            "content": """
                - You are an Korean English translator. 
                - Document is about Welfare service target.
                - You must organize the document including important information.
                - Output will be used for embedding with query, so you must translate it fit well to the queries possibly asked.
                - You only contain information inside the document.
                - Translation should start with "This document explains about the targets of the Welfare service. This service targets { explanation of service target }\".
                """},
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt_message,
        )
    except:
        return "Error"
    return response['choices'][0]['message']['content']


def get_translation_content(text):
    prompt_message = [
        {"role": "assistant",
            "content": """
                - You must translate the following sentence into English.
                - Output only translation, No prose or explanation, indicators.
                """},
        {"role": "user",
            "content": f"Translate {text} to english"
         },
        {"role": "system",
            "content": """
                - You are an Korean English translator. 
                - Document is about Welfare service contents.
                - You must organize the document including important information.
                - Output will be used for embedding with query, so you must translate it fit well to the queries possibly asked.
                - You only contain information inside the document.
                - Translation should start with "This document explains about the contents of the Welfare service, this service provides { contents of service }\" ".
                - Natural tone and short sentence is preferred.
                """},
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt_message,
        )
    except:
        return "Error"
    return response['choices'][0]['message']['content']


def get_embedding_df(df: pd.DataFrame):
    # Use progress_apply instead of apply
    df['content_embed'] = df['content'].progress_apply(lambda x: get_embedding([x]))
    df['target_embed'] = df['target'].progress_apply(lambda x: get_embedding([x]))
    return df


def get_embedding_df_from_parquet(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    return get_embedding_df(df)


def from_parquet_to_npy_array(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    zeros = np.zeros((len(df), len(df['content_embed'][0])))
    for i, row in df.iterrows():
        zeros[i] = row['content_embed']
    return zeros


def query_doc_embed():
    with open('preprocess/embedding/query_embed.json', 'r') as f:
        query_embed = json.load(f)

    data = []
    for key, value in query_embed.items():
        label = value['index']
        embed = value['embedding']
        query = value['query']
        data.append([label, query, embed])
    df = pd.DataFrame(data, columns=['index', 'query', 'embed'])
    df.to_parquet('preprocess/experiments/files/query_embed.parquet')


def sample_query_get_top_doc():
    df = pd.read_parquet('preprocess/experiments/files/query_embed.parquet')
    article_embed = pd.read_parquet('preprocess/experiments/files/articles_embed.parquet')
    file_title = pd.read_csv('preprocess/embedding/info_sheet.csv')

    for i, row in df.iterrows():
        if i % 10 != 0:
            continue
        query = row['query']
        embed = row['embed']

        article_embed['cosine_target'] = article_embed['target_embed'].apply(
            lambda x: np.dot(x, embed))
        article_embed['cosine_content'] = article_embed['content_embed'].apply(
            lambda x: np.dot(x, embed))

        article_embed = article_embed.sort_values(by=['cosine_target'], ascending=False)
        target_top_5 = file_title[file_title['filename'].isin(article_embed.head(5)['filename'])]['title']

        article_embed = article_embed.sort_values(by=['cosine_content'], ascending=False)
        content_top_5 = file_title[file_title['filename'].isin(article_embed.head(5)['filename'])]['title']

        print(f"query {i}: ", query)
        print("target: ")
        for title in target_top_5.values:
            print("      ", title)
        print("content: ")
        for title in content_top_5.values:
            print("      ", title)
        print()


def translate_to_english():
    df = pd.read_parquet('preprocess/experiments/files/articles_embed.parquet')
    start = time.time()
    for i, row in df.iterrows():
        if i % 10 == 0 and i != 0:
            print(f"[{i}/{len(df)}] {i/len(df) * 100}%, {time.time() - start}(s) {60 / (time.time() - start)}/min")
        content = row['content']
        target = row['target']
        content_eng = get_translation_content(content)
        target_eng = get_translation_target(target)

        if "Error" in content_eng or "Error" in target_eng:
            print(f"Error in {i}")

        df.loc[i, 'content_eng'] = content_eng
        df.loc[i, 'target_eng'] = target_eng

    df.to_parquet('preprocess/experiments/files/articles_eng.parquet')


if __name__ == '__main__':
    df = pd.read_parquet('preprocess/experiments/files/articles_eng.parquet')
    print(df.columns)

    missing = [1, 13, 183, ]
