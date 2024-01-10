from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

import json
import pandas as pd
import ast

from typing import List, Union

from apikey import load_env

OPENAI_API_KEY, SERPER_API_KEY = load_env()

def load_data_company(
        data_path : str = "data/custom/company_reviews.csv"
) -> List[Document]:
    df = pd.read_csv(data_path, encoding='utf-8')
    df.fillna("", inplace=True)
    documents = []

    for (_, row) in df.iterrows():
        review = row["name"] + " is a company working on" + row["industry"] + ". " + row["description"]
        specs = dict()
        specs.update(ast.literal_eval(row["happiness"]))
        specs.update(ast.literal_eval(row["ratings"]))
        specs.update(ast.literal_eval(row["roles"]))
        specs["industry"] = row["industry"]
        specs["name"] = row["name"]
        document = Document(
            page_content=review,
            metadata=specs
        )
        documents.append(document)

    return documents
    
def create_instance(
        embeddings : OpenAIEmbeddings, 
        path : str = "data/integrated/company_reviews"
) -> Chroma:
    instance = Chroma(embedding_function=embeddings, persist_directory=path)
    return instance

def ingrest_data(
        documents : Union[Document, List[Document]],
        instance : Chroma
) -> None:
    instance.add_documents(documents=documents[:100])
    instance.persist()
        

if __name__ == '__main__':
    documents = load_data_company()
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )
    chroma = create_instance(
        embeddings=embeddings
    )
    ingrest_data(
        documents=documents,
        instance=chroma
    )

    
    