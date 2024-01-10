import pinecone
from langchain.vectorstores import FAISS, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from utils import get_repo_name_from_url

def query_vdb_for_context_docs(query, pinecone_index_name, repo_url):
    repo_name = get_repo_name_from_url(repo_url)
    print(f"Querying VDB {repo_name}...")
    embeddings = OpenAIEmbeddings(disallowed_special=())
    pinecone_index = pinecone.Index(pinecone_index_name)
    pinecone_vdb = Pinecone(pinecone_index, embeddings.embed_query, "cakedocs", namespace=repo_name)
    context_docs = pinecone_vdb.similarity_search(query, k=4)
    print(f"Found {len(context_docs)} context docs")
    return context_docs
