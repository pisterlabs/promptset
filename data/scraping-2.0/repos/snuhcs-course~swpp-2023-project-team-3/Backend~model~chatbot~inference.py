import torch
from transformers import CLIPModel, CLIPProcessor
import os
import json
import numpy as np
import pandas as pd
import openai
from typing import List

cwd = os.getcwd()
model_dir = "model_config"

MODEL_PATH = os.path.join(cwd, model_dir)
FASHIONCLIP_SOURCE = "patrickjohncyh/fashion-clip"


class ClipTextEmbedding(object):
    model = None 
    processor = None
    
    @classmethod
    def _check_model(cls):
        return cls.model is not None

    @classmethod
    def __init__(cls):
        # if cls.model == None:
        #     print("It's None")
        cls.model, cls.processor = cls._get_model()

    @classmethod
    def _get_model(cls):
        if cls.model == None:
            if os.path.exists(MODEL_PATH) == False:
                cls.model = CLIPModel.from_pretrained(FASHIONCLIP_SOURCE)
                cls.processor = CLIPProcessor.from_pretrained(FASHIONCLIP_SOURCE)
                cls.model.save_pretrained(MODEL_PATH)
                cls.processor.save_pretrained(MODEL_PATH)
            else :
                cls.model = CLIPModel.from_pretrained(MODEL_PATH)
                cls.processor = CLIPProcessor.from_pretrained(MODEL_PATH)
        return cls.model, cls.processor

    @classmethod
    def encode_text(cls, inputText:List[str]):
        input_vectors = cls.processor(text=inputText, return_tensors="pt", max_length=77, padding="max_length", truncation=True)
        text_embeddings = cls.model.get_text_features(**input_vectors).detach().cpu().numpy()
        return text_embeddings
    
    @classmethod
    def load_imgVector(cls):
        # if os.path.exists("./img_vectors.csv") == False :
        #     print("no such file")
        #     download_imgvector()
        img_vector = np.loadtxt("./img_vectors.csv", delimiter=",")
        item_df = pd.read_csv("./item_df.csv")
        subset = item_df[['id']]
        return img_vector, subset

    @classmethod
    def get_similarity(cls, inputText:List[str])->pd.DataFrame:
        # Encode the query text to get a text embedding
        text_embedding = cls.encode_text(inputText)
        
        image_embeddings, subset = cls.load_imgVector()
        image_embeddings = image_embeddings/np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
        text_embedding = text_embedding/np.linalg.norm(text_embedding, ord=2, axis=-1, keepdims=True)
        similarities = np.dot(text_embedding, image_embeddings.T)
        
        for i in range(similarities.shape[0]):
            subset[f'query_{i}'] = similarities[i]

        itemdf = subset.copy()
        return itemdf

    @classmethod
    def ret_queries(cls, queryList:List[str]):
        itemdf = cls.get_similarity(queryList)
        finaldf = pd.DataFrame()
        dict_key = ["gpt_query1", "gpt_query2", "gpt_query3"]
        ret_dict = {}
        sendDict = {}
        finaldfs = []

        for i in range(len(queryList)):
            if (i<3):
                tmpdf = itemdf[["id", f'query_{i}']].copy()  # Make a copy to avoid SettingWithCopyWarning
                tmpdf['sim'] = tmpdf[f'query_{i}'].tolist()
                tmpdf['query_index'] = i
                tmpdf = tmpdf[["id", "sim", "query_index"]]
                finaldfs.append(tmpdf)
                sendDict[dict_key[i]] = queryList[i]
            else:
                break

        if len(sendDict.keys()) != 3 :
            nonKeys = set(dict_key).difference(set(sendDict.keys()))
            for key in nonKeys:
                sendDict[key] = None
        # Concatenate all DataFrames outside the loop
        finaldf = pd.concat(finaldfs, ignore_index=True)
        finaldf = finaldf.sort_values(by="sim", ascending=False)
        finaldf = finaldf.drop_duplicates(subset=["id"], keep="first")
        finaldf = finaldf.head(9)
        for i in range(len(finaldf)):
            ret_dict[int(finaldf.iloc[i]["id"])] = [finaldf.iloc[i]["sim"], finaldf.iloc[i]["query_index"]]
        return ret_dict, sendDict