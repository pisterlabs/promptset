import os
import json
import argparse
import time
import torch
import random
import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

from sklearn.cluster import DBSCAN, HDBSCAN, KMeans, Birch

from utils import load_jsonl, save_jsonl
from openai_utils import calculate_embeddings

def batchize(data, batch_size=2):
    idx, texts = [], []
    for d in data:
        idx.append(d['id'])
        texts.append(d['text'])
    batched_text = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    return idx, batched_text


def get_ada_embeddings(data, batch_size=64):
    idxes, batched_text = batchize(data, batch_size)

    embeddings = []

    for sentences in tqdm(batched_text):
        
        emb = calculate_embeddings(sentences)
        embeddings.extend(emb) 
        time.sleep(6)

    assert len(idxes) == len(embeddings), "id长度和embedding长度不一致，请检查embedding获取是否合理！"

    id_embeddings = []
    for idx, embedding in zip(idxes, embeddings):
        id_embeddings.append({
            "id": idx,
            "embedding": embedding.tolist()
        })
    return id_embeddings

def get_embeddings(data, batch_size=2):
    
    idxes, batched_text = batchize(data, batch_size)
    embeddings = []

    # ========= Model and Embedding =============
    model_name = "bert-base-uncased"
    
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
    
    model =AutoModel.from_pretrained(model_name, 
                                    device_map="auto",
                                    trust_remote_code=True,
                                    resume_download=True, 
                                    token="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").to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    

    for sentences in tqdm(batched_text):
        tokens = tokenizer.batch_encode_plus(
            sentences,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors='pt'
        )
        # Move tokenized inputs to device (e.g., GPU)
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state (B, T, H) 
        # TODO 取什么位置的表征
        emb = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy() # 取第一个字符
        embeddings.extend(emb) 
    # ===========================================

    assert len(idxes) == len(embeddings), "id长度和embedding长度不一致，请检查embedding获取是否合理！"

    id_embeddings = []
    for idx, embedding in zip(idxes, embeddings):
        id_embeddings.append({
            "id": idx,
            "embedding": embedding.tolist()
        })
    
    return id_embeddings


def get_cluster(id_embeddings, reduce_dim=None, c_algo="KMeans"):
    
    idxes, embeddings = [], []

    for d in id_embeddings:
        idxes.append(d['id'])
        embeddings.append(d['embedding'])
    embeddings = np.asarray(embeddings)
    
    # 降维，降维算法时间复杂度高，运行慢，所以最好运行一次之后存储好，之后直接加载即可
    if reduce_dim is not None:
        try:
            from umap import UMAP
        except Exception:
            from umap.umap_ import UMAP
        

        reducer = UMAP(n_neighbors=500, min_dist=0.1, n_components=reduce_dim, metric="euclidean", low_memory=False, n_jobs=-1)
        embeddings = reducer.fit_transform(embeddings)

    # 聚类
    if c_algo == "DBSCAN":
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
        # n_jobs 设置进程数量，-1使用所有的cpu
        clustering = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
    elif c_algo == "HDBSCAN":
        # 这个可以试一下好像效果更好
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
        clustering = HDBSCAN(n_jobs=-1)
    elif c_algo == "KMeans":
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
        clustering = KMeans(n_clusters=10, n_init="auto")
    elif c_algo == "Birch":
        clustering = Birch()
    elif c_algo == "AgglomerativeClustering":
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(n_clusters=10)
    elif c_algo == "BisectingKMeans":
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html#sklearn.cluster.BisectingKMeans
        from sklearn.cluster import BisectingKMeans
        clustering = BisectingKMeans(n_clusters=10)
    else:
        # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
        print("不支持其他聚类算法，请自行添加。")
    
    clustering.fit(embeddings)
    labels = clustering.labels_
    assert len(labels) == len(idxes)
    
    total_culsters = set(labels)
    print(f"聚成了 {len(total_culsters)} 类。")

    return idxes, labels, embeddings

def save_by_labels(data, labels, path):
    if not os.path.exists(path):
        os.makedirs(path)
    total_culsters = list(set(labels))

    saved_data = {}
    for label in total_culsters:
        saved_data[label] = []
    
    for d, label in zip(data, labels):
        saved_data[label].append(d)
    
    for k, v in saved_data.items():
        save_jsonl(v, os.path.join(path, f"{k}.jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_dir", default=None, help="输入文件所在文件夹")
    parser.add_argument("--in_file", default=None, help="输入文件名，{'id': xxx, 'text': xxx}")
    parser.add_argument("--out_dir", default=None, help="向量输出位置")
    parser.add_argument("--emb_method", default="bert", help="表征获取方法")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--reduce_dim", default=3, type=int)
    parser.add_argument("--cluster_algo", default="KMeans", help="建议使用：KMeans、DBSCAN、BisectingKMeans")

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    in_file_name = args.in_file.replace(".jsonl", "")
    out_file_name = f"{in_file_name}.{args.cluster_algo}.labels.tsv"
    out_emb_name = f"{in_file_name}.emb.jsonl"
    out_reduced_dim_name = f"{in_file_name}.emb.{str(args.reduce_dim)}.jsonl"

    # 加载原始文本数据
    data = load_jsonl(os.path.join(args.in_dir, args.in_file))
    
    # 如果存在降维后的表征，直接加载
    if os.path.exists(os.path.join(args.out_dir, out_reduced_dim_name)):
        id_embeddings = load_jsonl(os.path.join(args.out_dir, out_reduced_dim_name))
        args.reduce_dim = None # 还原成None，不需要进行降维操作了
    # 如果不存在降维后的表征，但是存在未降维的，直接加载
    elif os.path.exists(os.path.join(args.out_dir, out_emb_name)):
        id_embeddings = load_jsonl(os.path.join(args.out_dir, out_emb_name))
    else:
        if args.emb_method == "bert":
            id_embeddings = get_embeddings(data, args.batch_size)
            save_jsonl(id_embeddings, os.path.join(args.out_dir, out_emb_name))
        elif args.emb_method == "ada":
            id_embeddings = get_ada_embeddings(data, args.batch_size)
            save_jsonl(id_embeddings, os.path.join(args.out_dir, out_emb_name))

    idxes, labels, embeddings = get_cluster(id_embeddings, reduce_dim=args.reduce_dim, c_algo=args.cluster_algo)
    # 保存 labels
    pd.DataFrame({"id": idxes, "labels": labels}).to_csv(os.path.join(args.out_dir, out_file_name), sep="\t")

    if args.reduce_dim:
        # 保存降维后的表征
        reduced_embs = [{"id": x, "embedding": y.tolist()} for x,y in zip(idxes, embeddings)]
        save_jsonl(reduced_embs, os.path.join(args.out_dir, out_reduced_dim_name))

    # 按照类别保存到文件夹下不同文件中
    save_file_by_labels_path = os.path.join(args.out_dir, f"{in_file_name}-{args.cluster_algo}-grouped")
    save_by_labels(data, labels, save_file_by_labels_path)

    # TODO：可视化部分 有点小bug
    # 这里按照可视化的html内部写的格式转换，否则可视化的时候没有办法关联到文本数据。
    # visualizer.html line:25-31 可以按照自己数据形式改那边的代码也可以
    texts = ["<s>" + d['text'].replace("<|user|>:", "Human:").replace("\n\n<|bot|>:", "<|end_of_turn|>Assistant:") for d in data]
    visualizer_data = json.dumps({"text": texts, "embedding": embeddings.tolist(), "color": labels.tolist()})

    vis_out_file = f"{in_file_name}.{args.cluster_algo}.vis.json"
    with open(os.path.join(args.out_dir, vis_out_file), "w") as f:
        f.write(visualizer_data)
    import plotly.graph_objects as go
    # 3D Figure
    fig = go.Figure(data=[
        go.Scatter3d(
            x=embeddings[:, 0], y=embeddings[:, 1], z=embeddings[:, 2],
            mode="markers",
            marker=dict(
                size=1.5,
                opacity=0.8,
            )
        )
    ])

    fig.show()