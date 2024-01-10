import tqdm
from roformer import RoFormerTokenizer, RoFormerForCausalLM
import torch
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
from text2vec import SentenceModel
import openai


def eval_roformer_sim(hf_model_name, qd_df):
    tokenizer = RoFormerTokenizer.from_pretrained(hf_model_name)
    model = RoFormerForCausalLM.from_pretrained(hf_model_name)

    def compute_embedding(s):
        inputs = tokenizer(s, return_tensors="pt")
        with torch.no_grad():
            outputs = model.forward(**inputs)
            return outputs.pooler_output.cpu().numpy()[0]

    scores, labels = [], []
    for index, row in tqdm.tqdm(qd_df.iterrows()):
        q, d, label = row['q'], row['d'], row['label']
        v1 = compute_embedding(q)
        v2 = compute_embedding(d)
        cos_sim = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        scores.append(cos_sim)
        labels.append(label)
    cor = spearmanr(scores, labels).correlation
    print(f"{hf_model_name}: {cor}")


def eval_text2vec(hf_model_name, qd_df):
    model = SentenceModel(hf_model_name)
    scores, labels = [], []
    for index, row in tqdm.tqdm(qd_df.iterrows()):
        q, d, label = row['q'], row['d'], row['label']
        v1, v2 = model.encode([q, d])
        cos_sim = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        scores.append(cos_sim)
        labels.append(label)
    cor = spearmanr(scores, labels).correlation
    print(f"{hf_model_name}: {cor}")


def eval_openai_embedding(openai_model_id, qd_df):
    scores, labels = [], []
    for index, row in tqdm.tqdm(qd_df.iterrows()):
        q, d, label = row['q'], row['d'], row['label']
        v1 = openai.Embedding.create(input=q, model=openai_model_id)["data"][0]["embedding"]
        v2 = openai.Embedding.create(input=d, model=openai_model_id)["data"][0]["embedding"]
        v1, v2 = np.array(v1), np.array(v2)
        cos_sim = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        scores.append(cos_sim)
        labels.append(label)
    cor = spearmanr(scores, labels).correlation
    print(f"{openai_model_id}: {cor}")


if __name__ == '__main__':
    qd_df = pd.read_csv("data/qd.csv")

    eval_roformer_sim("junnyu/roformer_chinese_sim_char_small", qd_df)
    eval_roformer_sim("junnyu/roformer_chinese_sim_char_base", qd_df)
    eval_roformer_sim("junnyu/roformer_chinese_sim_char_ft_small", qd_df)
    eval_roformer_sim("junnyu/roformer_chinese_sim_char_ft_base", qd_df)

    eval_text2vec("shibing624/text2vec-base-chinese", qd_df)
    eval_text2vec("GanymedeNil/text2vec-base-chinese", qd_df)
    eval_text2vec("GanymedeNil/text2vec-large-chinese", qd_df)

    eval_openai_embedding("text-embedding-ada-002", qd_df)








