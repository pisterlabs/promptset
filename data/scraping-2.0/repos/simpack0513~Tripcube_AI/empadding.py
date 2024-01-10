# CSV 파일에 개요 부분을 임베딩하여 새로운 CSV 파일로 제작
import os
import pandas as pd
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def embedding(text):
	response = openai.Embedding.create(
		model="text-embedding-ada-002",
		input=text
	)
	return response["data"][0]["embedding"]

metadata = pd.read_csv("./fulldata.csv", sep=",")
metadata["embeddings"] = metadata["description"].apply(lambda x : embedding(x))

metadata.to_csv("./fulldata_embedding.csv", index=False,)
