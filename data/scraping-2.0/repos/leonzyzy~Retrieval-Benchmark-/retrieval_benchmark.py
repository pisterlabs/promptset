import re
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from metrics import *

# define a function to process the text files
def TextProcess(path, s, m):
    # load and process text files
    loader = DirectoryLoader(path, glob='*.txt', loader_cls=TextLoader)
    docs = loader.load()

    # splitting the text into
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=s, chunk_overlap=m)
    texts = text_splitter.split_documents(docs)

    # change metadata into int based on number 
    for text in texts:
        # extract the int from metadata
        text.metadata = int(re.findall(r'\d+', text.metadata['source'])[0])

    # group the texts by the source
    meta_sen = {}
    for i in range(0, len(docs)):
        sen = []
        for text in texts:
            if text.metadata == i:
                sen.append(text.page_content)
        meta_sen[i] = sen
    
    return meta_sen

# define a function to perform retrieval task
def RetrievalTask(queries, corpus, meta_sen, qrels, model, dist, k):
    # initialize metrics lists
    precision_k = []
    recall_k = []
    MRR_k = []
    MAP_k = []
    NDCG_K = []

    # get relevant corpus ids for each query
    rel_doc = {}
    for i in range(len(qrels)):
        if qrels['query-id'][i] not in rel_doc:
            rel_doc[qrels['query-id'][i]] = [qrels['corpus-id'][i]]
        else:
            rel_doc[qrels['query-id'][i]].append(qrels['corpus-id'][i])

    # get ids
    queries_ids = np.array(queries['_id'])
    queries_ids = queries_ids.astype(int)
    corpus_ids = np.array(corpus['_id'])
    corpus_ids = corpus_ids.astype(int)
    
    # encode queries and corpus
    queries_vecs = model.encode(queries['text'])
    meta_sen_vecs = {}
    for key in meta_sen:
        meta_sen_vecs[key] = model.encode(meta_sen[key])
    
    # compute distance/score for all queries and meta_sen_vecs
    meta_scores = {}
    for key in meta_sen_vecs:
        if dist == 'cosine':
            scores = cos_sim(queries_vecs, meta_sen_vecs[key])
        elif dist == 'dotprod':
            scores = dotprod(queries_vecs, meta_sen_vecs[key])
        elif dist == 'euclidean':
            scores = euclidean(queries_vecs, meta_sen_vecs[key]) 
        elif dist == 'chebyshev':
            scores = chebyshev(queries_vecs, meta_sen_vecs[key])

        # get maximum score for each query
        meta_scores[key] = torch.max(scores, dim=1).values.numpy().tolist()

    # combine dictionary to a 2D torch tensor
    meta_scores = torch.tensor(list(meta_scores.values())).T
    top_k_scores_idx = torch.topk(meta_scores, k, dim=1, 
                                         largest=True, sorted=False).indices.numpy()

    # get top k corpus ids for each query
    for i in range(len(queries_ids)):
        retrieved_list = [0]*k
        top_k_corpus_ids = corpus_ids[top_k_scores_idx[i]]
        if queries_ids[i] in rel_doc:
            rel_corpus_ids = rel_doc[queries_ids[i]]
            for j in range(k):
                if top_k_corpus_ids[j] in rel_corpus_ids:
                    retrieved_list[j] += 1
            
            # compute precision@k
            precision_k.append(sum(retrieved_list)/k)
            
            # compute recall@k
            recall_k.append(sum(retrieved_list)/len(rel_corpus_ids))
            
            # compute MRR@k
            for j in range(k):
                if retrieved_list[j] == 1:
                    MRR_k.append(1.0/(j+1))
                    break
                
            # compute MAP@k
            ap_k = []
            for j in range(k):
                if retrieved_list[j] == 1:
                    ap_k.append(sum(retrieved_list[:j+1])/(j+1))
            if sum(retrieved_list) != 0:
                MAP_k.append(sum(ap_k)/min(sum(retrieved_list), len(rel_corpus_ids)))
            else:
                MAP_k.append(0)
            
            # compute ndcg@k
            true_relavance = [1] * len(rel_corpus_ids)
            ndcg_k = compute_dcg_at_k(retrieved_list, k)/compute_dcg_at_k(true_relavance, k)
            NDCG_K.append(ndcg_k)
    
    # save metrics as a dictionary
    metrics = {'precision@k': sum(precision_k)/len(precision_k),
               'recall@k': sum(recall_k)/len(recall_k),
               'MRR@k': sum(MRR_k)/len(MRR_k)+0.0001,
               'MAP@k': sum(MAP_k)/len(MAP_k)+0.0001,
               'NDCG@k': sum(NDCG_K)/len(NDCG_K)}
        
    return metrics


# define parameters
path = './documents/'
s = 3000
m = 0
queries = load_dataset('BeIR/scifact', 'queries')['queries']
corpus = load_dataset('BeIR/scifact', 'corpus')['corpus']
qrels = pd.read_csv('./test.tsv', sep='\t')
model = SentenceTransformer('all-MiniLM-L6-v2')
meta_sen = TextProcess(path, s, m)
dist = 'chebyshev'
k = 10


results = RetrievalTask(queries, corpus, meta_sen, qrels, model, dist, k)
