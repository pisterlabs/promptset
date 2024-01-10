import os
import openai
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st

# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
# openai.api_key = OPENAI_API_KEY

openApiKey = st.secrets["OPENAI_API_KEY"]
# print(openApiKey)
openai.api_key = openApiKey

embedding_function = OpenAIEmbeddings()

def get_matches_resume(query, k=10, match_type="resume"):

    if match_type == "work":
        db = Chroma(persist_directory="chroma/work/",
                collection_name="resume_work",
                embedding_function=embedding_function)

    elif match_type == "skills":
        db = Chroma(persist_directory="chroma/skills/", 
                  collection_name="resume_skills",
                  embedding_function=embedding_function)
    else:
        db = Chroma(persist_directory="chroma/full_resume/",
                    collection_name="resume_full",
                    embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query, k=k)
    distance = [f"{result[1]:.2f}" for result in results]
    full_name = [result[0].metadata['full_name'] for result in results]
    content = [result[0].page_content for result in results]
    df = pd.DataFrame({"full_name": full_name, "distance": distance, "content": content})   
    return df

resumedb = Chroma(persist_directory="chroma/full_resume/",
                    collection_name="resume_full",
                    embedding_function=embedding_function)

def show_resume(full_name):
    full_resume = resumedb.get(where={"full_name": full_name})['documents'][0]
    return full_resume


