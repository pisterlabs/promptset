from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain

import numpy as np
from sklearn.cluster import KMeans

def resumeTranscript(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )

    docs = text_splitter.split_documents(data)

    if len(docs) > 4:
        vectors = createVectorStore(docs)
        docs = extractSentences(docs, vectors)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = load_summarize_chain(llm, chain_type="stuff")
    return chain.run(data)

def createVectorStore(docs):
    embeddings = OpenAIEmbeddings()
    vectors = embeddings.embed_documents([x.page_content for x in docs])

    return (vectors)

def extractSentences(docs, vectors):
    num_clusters = round(len(docs)*0.3)
    kmeans = KMeans(n_clusters=num_clusters).fit(vectors)

    closest_indices = []

    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis = 1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)
    
    selected_indices = sorted(closest_indices)
    selected_docs = [docs[doc] for doc in selected_indices]

    return selected_docs