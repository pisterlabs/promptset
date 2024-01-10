import pinecone
import streamlit as st
from langchain.vectorstores import Pinecone, DeepLake



# pinecone db 임베딩 후 리턴
# @st.cache_data()
def db_from_pinecone(docs, embeddings):
    
    # initialize pinecone
    pinecone.init(
        api_key= st.secrets["PINECONE_API_KEY"],
        environment= st.secrets["PINECONE_ENV"]  
    )
    index_name = st.secrets["PINECONE_INDEX_NAME"]
    
    # pinecone vector 삭제
    index = pinecone.Index(index_name)
    index.delete(deleteAll='true')
    vectorstore = Pinecone.from_documents(docs, embeddings, index_name=index_name)

    return vectorstore

def db_from_deeplake(docs, embeddings):
    # get it from https://app.activeloop.ai/
    user_name = st.secrets["DEEPLAKE_USERNAME"]
    file_name = st.secrets["ACTIVELOOP_FILE_NAME"]
    db = DeepLake.from_documents(
        docs, embeddings, dataset_path=f'hub://{user_name}/{file_name}'
    )

    db = DeepLake(
            dataset_path=f"hub://{user_name}/{file_name}",
            read_only=True,
            embedding_function=embeddings
    )
    return db

# @st.cache_data()
def mmr_retriever_setting(vectorstore, fetch_num, k_num):
    retriever = vectorstore.as_retriever(search_type="mmr")
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = fetch_num
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = k_num
    
    return retriever
