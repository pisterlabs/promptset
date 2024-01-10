from store.splitter import load_documents, load_urls
import openai
from dotenv import load_dotenv, find_dotenv
import os
from supabase import create_client, Client
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, SupabaseVectorStore
import pickle
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from termcolor import colored

def get_repo_names(dir_path):
    folder_names = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
    concatenated_names = "-".join(folder_names)
    return concatenated_names

class LocalHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_id="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_id)

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings

    def embed_query(self, text):
        embedding = self.model.encode(text)
        return list(map(float, embedding))


def create_vdb(knowledge, vdb_path=None):
    embedding_type = os.environ.get('EMBEDDING_TYPE', "local")
    if embedding_type == "local":
        embedding = LocalHuggingFaceEmbeddings()
    else:
        embedding = OpenAIEmbeddings(disallowed_special=())
    print(colored("Embedding documents...", "green"))
    faiss_store = FAISS.from_documents(knowledge["known_docs"], embedding=embedding)
    if vdb_path is not None:
        with open(vdb_path, "wb") as f:
            pickle.dump(faiss_store, f)
    return faiss_store


def get_vdb(vdb_path):
    with open(vdb_path, "rb") as f:
        faiss_store = pickle.load(f)
    return faiss_store


def supabase_vdb(knowledge):
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)

    vector_store = SupabaseVectorStore(client=supabase, embedding=OpenAIEmbeddings(), table_name="documents")
    vector_store.add_documents(knowledge["known_docs"])
    vector_store.add_texts(knowledge["known_text"]["pages"], metadatas=knowledge["known_text"]["metadatas"])

    return vector_store


def generate_knowledge_from_repo(dir_path, ignore_list):
    knowledge = {"known_docs": [], "known_text": {"pages": [], "metadatas": []}}
    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d not in ignore_list]  # modify dirs in-place
        for file in files:
            if file in ignore_list:
                continue
            filepath = os.path.join(root, file)
            try:
                # Using a more general way for code file parsing
                knowledge["known_docs"].extend(load_documents([filepath]))

            except Exception as e:
                print(f"Failed to process {filepath} due to error: {str(e)}")

    return knowledge


def get_or_create_knowledge_from_repo(dir_path="./code_repo"):
    vdb_path = "./vdb-" + get_repo_names(dir_path) + ".pkl"
    # check if vdb_path exists
    if os.path.isfile(vdb_path):
        print(colored("Local VDB found! Loading VDB from file...", "green"))
        vdb = get_vdb(vdb_path)
    else:
        print(colored("Generating VDB from repo...", "green"))
        ignore_list = ['.git', 'node_modules', '__pycache__', '.idea',
                       '.vscode']
        knowledge = generate_knowledge_from_repo(dir_path, ignore_list)
        vdb = create_vdb(knowledge, vdb_path=vdb_path)
    print(colored("VDB generated!", "green"))
    return vdb

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    openai.api_key = os.environ.get("OPENAI_API_KEY", "null")

    query = "What is the usage of this repo?"
    files = ["./README.md"]
    urls = ["https://github.com/JinghaoZhao/GPT-Code-Learner"]

    known_docs = load_documents(files)
    known_pages, metadatas = load_urls(urls)

    knowledge_base = {"known_docs": known_docs, "known_text": {"pages": known_pages, "metadatas": metadatas}}

    faiss_store = create_vdb(knowledge_base)
    matched_docs = faiss_store.similarity_search(query)
    for doc in matched_docs:
        print("------------------------\n", doc)

    supabase_store = supabase_vdb(knowledge_base)
    matched_docs = supabase_store.similarity_search(query)
    for doc in matched_docs:
        print("------------------------\n", doc)

    chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=faiss_store)
    result = chain({"question": query})
    print("FAISS result", result)

    chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(temperature=0), vectorstore=supabase_store)
    result = chain({"question": query})
    print("Supabase result", result)