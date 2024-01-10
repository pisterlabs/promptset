import numpy
import pandas
from langchain.embeddings.openai import OpenAIEmbeddings
from tqdm import tqdm


def get_embeddings(df):
    embedder = OpenAIEmbeddings()

    embed_list = []
    for text, label in tqdm(zip(df.text, df.label), total=len(df)):
        # total = f"{text.lower()} disease={label.lower()}"
        embed = embedder.embed_query(text)
        embed_list.append(embed)
    return embed_list


def get_2d_correlation_matrix(embed_list):
    sz = len(embed_list)
    cor2 = numpy.zeros((sz, sz))

    for i in range(sz):
        for j in range(i + 1, sz):
            cor2[i][j] = numpy.dot(embed_list[i], embed_list[j])
            cor2[j][i] = cor2[i][j]
    return cor2


def get_top_n_values_and_indices(data, top_n):
    top_indices = numpy.argsort(-data, axis=1)[:, :top_n]
    top_values = numpy.take_along_axis(data, top_indices, axis=1)
    return top_values, top_indices


def disease_finder_v1():
    df = pandas.read_csv("data/Symptom2Disease.csv")
    top_n = 3

    embed_list = get_embeddings(df)
    cor2 = get_2d_correlation_matrix(embed_list)
    _, top_indices = get_top_n_values_and_indices(cor2, top_n)

    labels = df.label.to_list()
    texts = df.text.to_list()

    cnt, cnt2, cnt3 = 0, 0, 0
    for idx, label in enumerate(labels):
        pred_labels = [labels[i] for i in top_indices[idx]]
        # top_texts = [texts[i] for i in top_indices[idx]]
        print(label, pred_labels)
        # print(texts[idx])
        # pprint(top_texts)
        # print("-"*100)
        cnt += int(label in pred_labels)
        cnt2 += int(pred_labels.count(label) >= 2)
        cnt3 += int(pred_labels.count(label) >= 3)

    print("at least one match", cnt / len(labels))
    print("at least two matches", cnt2 / len(labels))
    print("all matched", cnt3 / len(labels))
