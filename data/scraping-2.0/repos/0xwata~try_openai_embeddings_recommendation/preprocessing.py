import openai
import pandas as pd
import json

openai.api_key = [OPEN_AI_KEY]


# 入力用の文章をロード
with open("data/ideal_candidate_profile.json") as f:
    docs = json.load(f)

index = []
for doc in docs:
    # ここでベクトル化を行う
    # openai.embeddings_utils.embeddings_utilsを使うともっとシンプルにかけます
    res = openai.Embedding.create(model="text-embedding-ada-002", input=doc["strings"])

    # ベクトルをデータベースに追加
    index.append({"strings": doc["strings"], "embedding": res["data"][0]["embedding"]})

with open("data/ideal_candidate_profile_index.json", "w") as f:
    json.dump(index, f, ensure_ascii=False)
