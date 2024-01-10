import  pandas as pd
import numpy as np
import openai
import os
from openai.embeddings_utils import  cosine_similarity, get_embedding
from sklearn.metrics import classification_report
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
datafile_path = "/Users/zm/aigcData/withEmbedding.csv"

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(eval).apply(np.array)

def evaluate_embeddings_approach(
        labels = ['negative', 'positive'],
        model = 'text-similarity-davinci-001',
):
    label_embeddings = [get_embedding(label, engine=model) for label in labels]

    def label_score(review_embedding, label_embeddings):
        return cosine_similarity(review_embedding, label_embeddings[1]) - cosine_similarity(review_embedding, label_embeddings[0])

    probas = df["embedding"].apply(lambda x: label_score(x, label_embeddings))
    preds = probas.apply(lambda x: 'positive' if x>0 else 'negative')

    report = classification_report(df.sentiment, preds)
    print(report)

    display = PrecisionRecallDisplay.from_predictions(df.sentiment, probas, pos_label='positive')
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    display.plot()
    plt.show()
evaluate_embeddings_approach(labels=['An Amazon review with a negative sentiment.','An Amazon review with a positive sentiment.'])