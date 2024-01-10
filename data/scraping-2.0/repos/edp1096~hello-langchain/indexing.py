from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Annoy
from langchain.document_loaders import TextLoader

embeddings = LlamaCppEmbeddings(model_path="./vicuna-7B-1.1-ggml_q4_0-ggjt_v3.bin", n_ctx=2048)

from langchain.document_loaders import TextLoader

loader = TextLoader("./training_full_clean.txt")
loader.encoding = "utf-8"
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=len)
docs = text_splitter.split_documents(documents)


# db = Annoy.from_documents(docs, embeddings)
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

# Now let's test it out
query = "Who is Whitfield Diffie?"
docs = db.similarity_search(query)
for doc in docs:
    print(doc.page_content)
