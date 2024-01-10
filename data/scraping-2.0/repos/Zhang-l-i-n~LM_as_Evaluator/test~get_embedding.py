import json
import openai
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
import numpy as np
import umap.plot
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation


def show_cluster(n_clusters, X, y, label):
    cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    y_pred = cluster.labels_  # 获取训练后对象的每个样本的标签
    centroid = cluster.cluster_centers_
    color = ['red', 'pink', 'orange', 'gray']
    fig, axi1 = plt.subplots(1)
    for i in range(n_clusters):
        axi1.scatter(X[y_pred == i, 0], X[y_pred == i, 1],
                     marker='o',
                     s=8,
                     c=color[i])
    axi1.scatter(centroid[:, 0], centroid[:, 1], marker='x', s=100, c='black')
    plt.show()


openai.api_version = '2023-03-15-preview'
openai.api_type = 'azure'
openai.api_base = "https://conmmunity-openai.openai.azure.com/"
openai.api_key = '3371b75d06a54deabcdd5818629ca833'


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, engine="ada-1"):
    return openai.Embedding.create(engine=engine, input=[text])["data"][0]["embedding"]


def embeddings(f_txt):
    embedding_list = []
    count = 0
    with open(f_txt, 'r', encoding='utf-8') as frd:
        for line in frd:
            count = count + 1

    pbar = tqdm(total=count)
    with open(f_txt, 'r', encoding='utf-8') as f:
        line = f.readline()
        # type = 0
        while line:

            if len(line) > 1:
                if line.startswith('    '):
                    level = 1
                elif line.strip():
                    level = 0
                    # type += 1
                embedding_list.append(
                    {"label": line.strip(),
                     "level": level,
                     # "type": type,
                     "type": 1,
                     "embedding": get_embedding(line)
                     }
                )
            line = f.readline()
            pbar.update(1)
    pbar.close()

    return embedding_list


def get_data(f_data, y='level'):
    embedding_list, label_list, y_list = [], [], []
    data_json = json.load(open(f_data, 'r', encoding='utf-8'))
    for d in data_json:
        embedding_list.append(d['embedding'])
        label_list.append(d['label'])
        y_list.append(d[y])
    # label_list = eval(open(f_label, 'r').readline().strip())
    return embedding_list, label_list, y_list


def data_prepare(f_raw, f_emb):
    embedding_list = embeddings(f_raw)
    with open(f_emb, 'w', encoding='utf=8') as f:
        f.write(json.dumps(embedding_list, ensure_ascii=False))
    # label_list = []
    # with open(f_raw, 'r', encoding='utf-8') as f:
    #     line = f.readline()
    #     while line:
    #         if line.strip():
    #             label_list.append(line.strip())
    #         line = f.readline()
    # open(f_label, 'w', encoding='utf=8').write(str(label_list))


if __name__ == '__main__':
    f_raw, f_emb = '计算机学科.level', 'embedding_bio2.json'

    ############
    # print('data preparing...')
    # data_prepare(f_raw, f_emb)
    # print('data prepared')
    # exit()

    font_path = '/Users/zllll/Library/Fonts/SimHei.ttf'
    custom_font = FontProperties(fname=font_path)

    embedding_list, label_list, y_list = get_data(f_emb, 'type')

    # embedding_list = eval(open(f_emb, 'r').readline().strip())
    print(len(embedding_list))
    X = np.array(embedding_list)
    y = np.array([1] * len(embedding_list))

    ################
    # UMAP降维
    ################
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(X)
    print(embedding.shape)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=y_list, cmap='rainbow')
    # 添加标签
    for i in range(len(label_list)):
        plt.text(embedding[:, 0][i], embedding[:, 1][i], label_list[i], ha='center', va='bottom',
                 fontproperties=custom_font, fontsize=5)

    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP')
    plt.show()

    ################
    # Affinity Propagation
    ################
    af = AffinityPropagation(preference=None).fit(embedding)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters = len(cluster_centers_indices)

    color = sns.color_palette("hls", n_clusters)
    for k, col in zip(range(n_clusters), color):
        class_members = labels == k
        cluster_center = embedding[cluster_centers_indices[k]]
        plt.scatter(embedding[class_members, 0], embedding[class_members, 1],
                    marker='o',
                    s=8,
                    c=col)
        plt.scatter(cluster_center[0], cluster_center[1], marker='x', s=100, c='black')

    for i in range(len(label_list)):
        plt.text(embedding[:, 0][i], embedding[:, 1][i], label_list[i], ha='center', va='bottom',
                 fontproperties=custom_font, fontsize=5)

    plt.title('AP--Estimated number of clusters: %d' % n_clusters)
    plt.show()

    ################
    # Kmeans聚类（UMAP降维后的数据）
    ################
    cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(embedding)
    y_pred = cluster.labels_  # 获取训练后对象的每个样本的标签
    centroid = cluster.cluster_centers_
    print(y_pred)
    print(centroid)
    color = sns.color_palette("hls", n_clusters)
    for i in range(n_clusters):
        plt.scatter(embedding[y_pred == i, 0], embedding[y_pred == i, 1],
                     marker='o',
                     s=8,
                     c=color[i])
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', s=100, c='black')
    for i in range(len(label_list)):
        plt.text(embedding[:, 0][i], embedding[:, 1][i], label_list[i], ha='center', va='bottom',
                 fontproperties=custom_font, fontsize=5)
    plt.title('Kmeans')
    plt.show()

    ################
    # Affinity Propagation（UMAP降维前的数据）
    ################
    af = AffinityPropagation(preference=None).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters = len(cluster_centers_indices)
    # print(X.shape)
    # print(cluster_centers_indices)

    X_expand_2d = reducer.fit_transform(X)

    color = sns.color_palette("hls", n_clusters)
    for k, col in zip(range(n_clusters), color):
        class_members = labels == k
        cluster_center = X_expand_2d[cluster_centers_indices[k]]
        plt.scatter(X_expand_2d[class_members, 0], X_expand_2d[class_members, 1],
                    marker='o',
                    s=8,
                    c=col)
        plt.scatter(cluster_center[0], cluster_center[1], marker='x', s=100, c='black')

    for i in range(len(label_list)):
        plt.text(X_expand_2d[:, 0][i], X_expand_2d[:, 1][i], label_list[i], ha='center', va='bottom',
                 fontproperties=custom_font, fontsize=5)

    plt.title('AP--Estimated number of clusters: %d' % n_clusters)
    plt.show()

    ################
    # Kmeans聚类（UMAP降维前的数据）
    ################
    cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    y_pred = cluster.labels_  # 获取训练后对象的每个样本的标签
    centroid = cluster.cluster_centers_
    # print(X)
    # print(centroid)
    X_expand = np.concatenate([X, centroid], axis=0)
    X_expand_2d = reducer.fit_transform(X_expand)
    embedding = X_expand_2d[:X.shape[0]]
    centroid = X_expand_2d[X.shape[0]:]
    color = sns.color_palette("hls", n_clusters)
    for i in range(n_clusters):
        plt.scatter(embedding[y_pred == i, 0], embedding[y_pred == i, 1],
                     marker='o',
                     s=8,
                     c=color[i])
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', s=100, c='black')
    for i in range(len(label_list)):
        plt.text(embedding[:, 0][i], embedding[:, 1][i], label_list[i], ha='center', va='bottom',
                 fontproperties=custom_font, fontsize=5)
    plt.title('Kmeans')
    plt.show()

