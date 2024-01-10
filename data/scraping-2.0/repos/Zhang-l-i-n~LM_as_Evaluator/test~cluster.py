import json
import time

import openai
from scipy.cluster import hierarchy
from tqdm import tqdm
import numpy as np
import umap.plot
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram


def get_data(f_data, feature='level'):
    embedding_list, label_list, y_list = [], [], []
    data_json = json.load(open(f_data, 'r', encoding='utf-8'))
    for d in data_json:
        # if d[feature] == 2:
        embedding_list.append(d['embedding'])
        label_list.append(d['label'])
        y_list.append(d[feature])
    # label_list = eval(open(f_label, 'r').readline().strip())
    return embedding_list, label_list, y_list


if __name__ == '__main__':
    plt.margins(0, 0)
    class_name = 'roberta_history'
    f_emb = '../data/embedding/chinese-roberta-wwm-ext-large_eb_history_0814.json'

    font_path = '/Users/zhanglin/Library/Fonts/SimHei.ttf'
    custom_font = FontProperties(fname=font_path)

    embedding_list, label_list, y_list = get_data(f_emb, 'type')

    # embedding_list = eval(open(f_emb, 'r').readline().strip())
    print(len(embedding_list))
    X = np.array(embedding_list)
    y = np.array([1] * len(embedding_list))

    ################
    # UMAP降维
    ################
    reducer = umap.UMAP(random_state=60)
    embedding = reducer.fit_transform(X)

    print(embedding.shape)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=y_list, cmap='rainbow')
    # 添加标签
    for i in range(len(label_list)):
        plt.text(embedding[:, 0][i], embedding[:, 1][i], label_list[i], ha='center', va='bottom',
                 fontproperties=custom_font, fontsize=12)

    plt.gca().set_aspect('equal', 'datalim')
    plt.title('')
    plt.savefig(class_name + '_a_UMAP_fig-time{}.png'.format(time.strftime("%Y%m%d-%H%M", time.localtime())),
                dpi=500)
    plt.show()
    ################
    # Affinity Propagation（降维后的数据）
    ################
    # af = AffinityPropagation(preference=None).fit(embedding)
    # cluster_centers_indices = af.cluster_centers_indices_
    # labels = af.labels_
    # n_clusters = len(cluster_centers_indices)
    #
    # color = sns.color_palette("hls", n_clusters)
    # for k, col in zip(range(n_clusters), color):
    #     class_members = labels == k
    #     cluster_center = embedding[cluster_centers_indices[k]]
    #     plt.scatter(embedding[class_members, 0], embedding[class_members, 1],
    #                 marker='o',
    #                 s=8,
    #                 c=col)
    #     plt.scatter(cluster_center[0], cluster_center[1], marker='x', s=100, c='black')
    #     for i in range(len(label_list)):
    #         if embedding[:, 0][i] == cluster_center[0] and embedding[:, 1][i] == cluster_center[1]:
    #             print(label_list[i])
    #
    # for i in range(len(label_list)):
    #     plt.text(embedding[:, 0][i], embedding[:, 1][i], label_list[i], ha='center', va='bottom',
    #              fontproperties=custom_font, fontsize=5)
    #
    # plt.title('AP-after_reducer--clusters: %d' % n_clusters)
    # plt.savefig(class_name + '_a_AP_fig-time{}.png'.format(time.strftime("%Y%m%d-%H%M", time.localtime())),
    #             dpi=500)
    # plt.show()

    # ################
    # # DBSCAN（UMAP降维后的数据）
    # ################
    # cluster = DBSCAN(min_samples=3).fit(embedding)
    # y_pred = cluster.labels_  # 获取训练后对象的每个样本的标签
    # clusters = {}
    # i = 0
    # for y in y_pred:
    #     if y not in clusters:
    #         clusters[y] = i
    #         i += 1
    # color = sns.color_palette("hls", len(clusters))
    # for y in clusters.keys():
    #     plt.scatter(embedding[y_pred == y, 0], embedding[y_pred == y, 1],
    #                 marker='o',
    #                 s=8,
    #                 c=color[y])
    # for i in range(len(label_list)):
    #     plt.text(embedding[:, 0][i], embedding[:, 1][i], label_list[i], ha='center', va='bottom',
    #              fontproperties=custom_font, fontsize=5)
    # plt.title('DBSCAN-after_reducer')
    # plt.savefig(class_name + '_a_DBSCAN_fig-time{}.png'.format(time.strftime("%Y%m%d-%H%M", time.localtime())),
    #             dpi=500)
    # plt.show()
    #
    # ################
    # # Agglomerative（UMAP降维后的数据）
    # ################
    # cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward').fit(embedding)
    # y_pred = cluster.labels_
    # plt.scatter(embedding[:,0],embedding[:,1], c=cluster.labels_, cmap='rainbow')
    # for i in range(len(label_list)):
    #     plt.text(embedding[:, 0][i], embedding[:, 1][i], label_list[i], ha='center', va='bottom',
    #              fontproperties=custom_font, fontsize=5)
    # plt.title('Agglomerative-after_reducer')
    # plt.savefig(class_name + '_a_Agglomerative_fig-time{}.png'.format(time.strftime("%Y%m%d-%H%M", time.localtime())),
    #             dpi=500)
    # plt.show()
    # # # 绘制聚类树
    # # Z = hierarchy.linkage(cluster.children_, method='ward')
    # # fig = plt.figure(figsize=(10, 5))
    # # dn = hierarchy.dendrogram(Z)
    # # plt.show()
    #
    # ################
    # # Affinity Propagation（UMAP降维前的数据）
    # ################
    # af = AffinityPropagation(preference=None).fit(X)
    # cluster_centers_indices = af.cluster_centers_indices_
    # labels = af.labels_
    # n_clusters = len(cluster_centers_indices)
    # # print(X.shape)
    # # print(cluster_centers_indices)
    # X_expand_2d = reducer.fit_transform(X)
    #
    # color = sns.color_palette("hls", n_clusters)
    # for k, col in zip(range(n_clusters), color):
    #     class_members = labels == k
    #     cluster_center = X_expand_2d[cluster_centers_indices[k]]
    #     plt.scatter(X_expand_2d[class_members, 0], X_expand_2d[class_members, 1],
    #                 marker='o',
    #                 s=8,
    #                 c=col)
    #     plt.scatter(cluster_center[0], cluster_center[1], marker='x', s=100, c='black')
    #
    # for i in range(len(label_list)):
    #     plt.text(X_expand_2d[:, 0][i], X_expand_2d[:, 1][i], label_list[i], ha='center', va='bottom',
    #              fontproperties=custom_font, fontsize=5)
    # plt.title('AP-before_reducer--clusters: %d' % n_clusters)
    # plt.savefig(class_name + '_b_AP_fig-time{}.png'.format(time.strftime("%Y%m%d-%H%M", time.localtime())),
    #             dpi=500)
    # plt.show()
    #
    # ################
    # # DBSCAN（UMAP降维前的数据）
    # ################
    # cluster = DBSCAN(min_samples=3).fit(X)
    # embedding = reducer.fit_transform(X)
    # y_pred = cluster.labels_  # 获取训练后对象的每个样本的标签
    # clusters = {}
    # i = 0
    # for y in y_pred:
    #     if y not in clusters:
    #         clusters[y] = i
    #         i += 1
    # color = sns.color_palette("hls", len(clusters))
    # for y in clusters.keys():
    #     plt.scatter(embedding[y_pred == y, 0], embedding[y_pred == y, 1],
    #                 marker='o',
    #                 s=8,
    #                 c=color[y])
    # for i in range(len(label_list)):
    #     plt.text(embedding[:, 0][i], embedding[:, 1][i], label_list[i], ha='center', va='bottom',
    #              fontproperties=custom_font, fontsize=5)
    # plt.title('DBSCAN-before_reducer')
    # plt.savefig(class_name + '_b_DBSCAN_fig-time{}.png'.format(time.strftime("%Y%m%d-%H%M", time.localtime())),
    #             dpi=500)
    # plt.show()
    #
    # ################
    # # Agglomerative（UMAP降维前的数据）
    # ################
    # cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward').fit(X)
    # embedding = reducer.fit_transform(X)
    # y_pred = cluster.labels_
    # plt.scatter(embedding[:,0],embedding[:,1], c=cluster.labels_, cmap='rainbow')
    # for i in range(len(label_list)):
    #     plt.text(embedding[:, 0][i], embedding[:, 1][i], label_list[i], ha='center', va='bottom',
    #              fontproperties=custom_font, fontsize=5)
    # plt.title('Agglomerative-before_reducer')
    # plt.savefig(class_name + '_b_Agglomerative_fig-time{}.png'.format(time.strftime("%Y%m%d-%H%M", time.localtime())),
    #             dpi=500)
    # plt.show()
