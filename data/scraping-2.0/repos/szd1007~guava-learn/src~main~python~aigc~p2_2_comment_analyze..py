import  pandas as pd
import numpy as np
import openai
import os
from openai.embeddings_utils import  cosine_similarity, get_embedding
from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay

datafile_path = "/Users/zm/aigcData/AllProductReviews.csv"

#todo 全量数据太费钱，先限制100条跑一下
#todo 模型不同，成本不同，这个到时候改为  text-embedding-ada-002   （最小的ada模型）
df = pd.read_csv(datafile_path)
df = df[df.ReviewStar != 3]
df["sentiment"] = df.ReviewStar.replace({1: "negative", 2:"negative", 4: "positive", 5: "positive"})
count = 1
df["embedding"] =""
#
for index, row in df.iterrows():
    df.loc[index, "embedding"] = str(get_embedding(row["ReviewBody"]))
    count += 1
    print("请求次数 %s:" % count)
    if count > 100:
        break

outPd = df[df.embedding != ""]

outPd.to_csv("/Users/zm/aigcData/withEmbedding.csv")