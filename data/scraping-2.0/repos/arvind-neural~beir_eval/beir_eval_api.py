'''
Intsall 
OpenAI: https://beta.openai.com/docs/developer-quickstart/python-bindings
GPT Tokenizer: https://huggingface.co/docs/transformers/model_doc/gpt2#gpt2tokenizerfast
BEIR benchmark: https://github.com/UKPLab/beir
'''
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import openai, os, numpy as np
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
from typing import List, Dict
openai.api_key = os.getenv("OPENAI_API_KEY")

def check_length(txt):
    tokens = tokenizer(txt)["input_ids"]
    if len(tokens) < 2040:
        return True
    return False

def truncate(txt):
    max_len = 1000
    while check_length(txt) == False:
        txt = " ".join(txt.split(" ")[:max_len])
        max_len -= 100
    return txt
    
class CustomDEModel:
    def __init__(self, **kwargs):
        self.model_name = kwargs["model_name"]

    def get_emb(self, txt, input_type):
        engine_name = self.model_name + "-query-001" if input_type == "query" else self.model_name + "-doc-001"
        txt = truncate(txt)
        response = openai.Embedding.create(input=txt, engine=engine_name)
        embeddings = response['data'][0]['embedding']
        return embeddings

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        embeddings = []
        print("encoding queries")
        for query in queries:
            emb = self.get_emb(query, "query")
            embeddings.append(emb)
        embeddings = np.vstack(embeddings)
        return embeddings

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        embeddings = []
        print(len(corpus))
        print("encoding docs")
        for docs in corpus:
            doc = docs["text"]
            emb = self.get_emb(doc, "doc")
            embeddings.append(emb)
        embeddings = np.vstack(embeddings)
        print("Embeddings shape:", embeddings.shape)
        return embeddings

def beir_search(model_name):
    custom_model = DRES(CustomDEModel(model_name=model_name))
    retriever = EvaluateRetrieval(custom_model, score_function="cos_sim")
    datasets = ["fiqa"]

    avg = 0.0
    for dataset in datasets:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
            dataset
        )
        print(url)
        data_path = util.download_and_unzip(url, "datasets")
        print(data_path)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        print(len(corpus), len(queries), len(qrels))
    
        results = retriever.retrieve(corpus, queries)
        #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
        
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        print(dataset, " NDCG@10: ", ndcg["NDCG@10"])
        avg += ndcg["NDCG@10"]
    print("Average NDCG@10: ", avg / len(datasets))

model_name = "text-search-ada" # "text-search-ada" or "text-search-babbage" or "text-search-curie" or "text-search-davinci"
beir_search(model_name)
