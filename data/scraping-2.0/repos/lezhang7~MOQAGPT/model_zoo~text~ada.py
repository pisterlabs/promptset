
from typing import List,Dict,Any,Union,Tuple
from tqdm import tqdm
import os
import openai
import torch
import json
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's best embeddings as of Apr 2023
BATCH_SIZE = 1000
MAX_LEN = 8191

class ADA_FOR_FEATURE_EXTRACTION():
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name='ada'
    def extract_text_features(self,document_ref:List[Tuple[str, str]],bs=2048,save_path=None):
        """
        document_ref: List of tuples (document_id, document_text)
        """
        document_dict={}
        for i in tqdm(range(len(document_ref)//bs+1)):
            document_ref_batch=document_ref[i*bs:(i+1)*bs]
            document_batch=[]
            document_ids=[]
            for document_id, document_text in document_ref_batch:
                document_batch.append(document_text)
                document_ids.append(document_id)
            response = openai.Embedding.create(model=EMBEDDING_MODEL, input=document_batch)
            for i, be in enumerate(response["data"]):
                assert i == be["index"]  # double check embeddings are in same order as input
            batch_embeddings = [e["embedding"] for e in response["data"]]
            for j in range(len(document_ids)):
                document_dict[document_ids[j]]=torch.tensor(batch_embeddings[j])

        if save_path is not None:
            torch.save(document_dict,save_path)
        return document_dict
