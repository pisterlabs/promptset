import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import SentenceTransformerEmbeddings

pinecone.init(api_key='Pinecone api key', environment='Pinecone environment')
embeddings = SentenceTransformerEmbeddings(model_name="google/flan-t5-base") #Use embedding from encoder model
index = Pinecone.from_existing_index(index_name="Pinecone index name", embedding=embeddings)


def get_similiar_docs(query, k=2, score=False):
    if score:
        similar_docs = index.similarity_search(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

query = "Which documents do I need for an Aadhar card?" #Query for similarity search
similar_docs = get_similiar_docs(query)
print(similar_docs)