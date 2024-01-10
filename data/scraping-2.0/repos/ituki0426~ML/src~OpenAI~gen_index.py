import openai
import os

import json

# 入力用の文章をロード
with open('./docs.json') as f:
    docs = json.load(f)

index = []
for doc in docs:
    # ここでベクトル化を行う
    # openai.embeddings_utils.embeddings_utilsを使うともっとシンプルにかけます
    res = openai.Embedding.create(
        model='text-embedding-ada-002',
        input=doc['title']
    )

    # ベクトルをデータベースに追加
    index.append({
        'title': doc['title'],
        'embedding': res['data'][0]['embedding']
    })

with open('./index.json', 'w') as f:
    json.dump(index, f,ensure_ascii=False)