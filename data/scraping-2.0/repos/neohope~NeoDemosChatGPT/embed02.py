#!/usr/bin/env python3
# -*- coding utf-8 -*-


import pandas as pd
import numpy as np
import openai
import yaml
from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay
from openai.embeddings_utils import cosine_similarity, get_embedding


'''
提醒：批量数据调用，请做好流量控制

通过pandas读入评价内容
通过get_embedding获得向量，通过label_score计算分值
通过sklearn进行分类，并展示结果
'''

# 选择使用最小的ada模型
EMBEDDING_MODEL = "text-embedding-ada-002"

# 获取访问open ai的密钥
def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


def label_score(review_embedding, label_embeddings):
    return cosine_similarity(review_embedding, label_embeddings[1]) - cosine_similarity(review_embedding, label_embeddings[0])


def evaluate_embeddings_approach(df, labels = ['negative', 'positive'],  model = EMBEDDING_MODEL,):
    label_embeddings = [get_embedding(label, engine=model) for label in labels]

    probas = df["embedding"].apply(lambda x: label_score(x, label_embeddings))
    preds = probas.apply(lambda x: 'positive' if x>0 else 'negative')

    report = classification_report(df.sentiment, preds)
    print(report)

    display = PrecisionRecallDisplay.from_predictions(df.sentiment, probas, pos_label='positive')
    _ = display.ax_.set_title("2-class Precision-Recall curve")


if __name__ == '__main__':
    get_api_key()

    # csv to dataframe
    datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"
    df = pd.read_csv(datafile_path)
    df["embedding"] = df.embedding.apply(eval).apply(np.array)

    # convert 5-star rating to binary sentiment
    # 去掉了3三星评论
    df = df[df.Score != 3]
    df["sentiment"] = df.Score.replace({1: "negative", 2: "negative", 4: "positive", 5: "positive"})

    evaluate_embeddings_approach(df, labels=['An Amazon review with a negative sentiment.', 'An Amazon review with a positive sentiment.'])

