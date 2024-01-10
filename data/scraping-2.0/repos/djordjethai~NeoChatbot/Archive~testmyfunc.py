import streamlit as st

from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate)

from myfunc.mojafunkcija import open_file
from pinecone_text.sparse import BM25Encoder
import os
from streamlit_javascript import st_javascript
import pandas as pd
from io import StringIO
from ast import literal_eval

import pinecone
import openai

client = openai.OpenAI()


def read_aad_username():
    js_code = """(await fetch("/.auth/me")
        .then(function(response) {return response.json();}).then(function(body) {return body;}))
    """

    return_value = st_javascript(js_code)

    username = None
    if return_value == 0:
        pass  # the result before the actual value is returned
    elif isinstance(return_value, list) and len(return_value) > 0:  # the actual value
        username = return_value[0]["user_id"]
    else:
        st.warning(
            f"could not directly read username from azure active directory: {return_value}.")
    
    return username
    

def azure_load_or_upload(blob_service_client, data, x):
    if x == "Load":
        try:
            streamdownloader = blob_service_client.get_container_client(
                "positive-user").get_blob_client("assistant_data.csv").download_blob()
            
            df = pd.read_csv(StringIO(streamdownloader.readall().decode("utf-8")), 
                             usecols=["user", "chat", "ID", "assistant", "fajlovi"])
            
            df["fajlovi"] = df["fajlovi"].apply(literal_eval)
            return df.dropna(how="all")
                     
        except FileNotFoundError:
            return {"Nisam pronasao fajl"}
        except Exception as e:
            return {f"An error occurred: {e}"}
    
    else:
        data["fajlovi"] = data["fajlovi"].apply(lambda data: str(data))     # konvertujemo listu u string
        blob_client = blob_service_client.get_blob_client("positive-user", "assistant_data.csv")
        blob_client.upload_blob(data.to_csv(index=False), overwrite=True)


def hybrid_search_process(upit: str) -> str:
        alpha = 0.5

        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY_POS"],
            environment=os.environ["PINECONE_ENVIRONMENT_POS"],
        )
        index = pinecone.Index("positive")

        def hybrid_query():
            def get_embedding(text, model="text-embedding-ada-002"):
                text = text.replace("\n", " ")
                return client.embeddings.create(input = [text], model=model).data[0].embedding
        
            hybrid_score_norm = (lambda dense, sparse, alpha: 
                                 ([v * alpha for v in dense], 
                                  {"indices": sparse["indices"], 
                                   "values": [v * (1 - alpha) for v in sparse["values"]]}
                                   ))
            hdense, hsparse = hybrid_score_norm(
                sparse = BM25Encoder().fit([upit]).encode_queries(upit),
                dense=get_embedding(upit),
                alpha=alpha,
            )
            return index.query(
                top_k=6,
                vector=hdense,
                sparse_vector=hsparse,
                include_metadata=True,
                namespace="zapisnik",
                ).to_dict()

        tematika = hybrid_query()

        uk_teme = ""
        for _, item in enumerate(tematika["matches"]):
            if item["score"] > 0.05:    # score
                uk_teme += item["metadata"]["context"] + "\n\n"

        system_message = SystemMessagePromptTemplate.from_template(
            template="You are a helpful assistent. You always answer in the Serbian language.").format()

        human_message = HumanMessagePromptTemplate.from_template(
            template=open_file("prompt_FT.txt")).format(
                zahtev=upit,
                uk_teme=uk_teme,
                ft_model="gpt-4-1106-preview",
                )
        return str(ChatPromptTemplate(messages=[system_message, human_message]))



def upload_to_openai(filepath):
    with open(filepath, "rb") as f:
        response = openai.files.create(file=f.read(), purpose="assistants")
    return response.id