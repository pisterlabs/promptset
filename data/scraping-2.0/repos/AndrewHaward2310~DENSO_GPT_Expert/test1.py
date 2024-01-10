import os
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Weaviate
model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
#database = Chroma(persist_directory="./chroma_db", embedding_function=model)

_ = load_dotenv(find_dotenv(), override=True)

OPEN_AI_API_KEY=os.environ['OPEN_AI_API_KEY']

def split_document(docs, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    # Splitting the documents into chunks
    chunks = text_splitter.create_documents([docs])
    return chunks


def insert_pdf_to_db(file_path):
    # Load pdf into pages
    pages = fitz.open(file_path)
    chunks = []  # create empty chunks
    # insert từng chunk vào chunk
    for page in pages:
        docs = split_document(page.get_text().replace('\n', ' ').lower())  # Return Langchain Documents list

        for doc in docs:
            chunk = Document(page_content=doc.page_content, metadata={'source': pages.name, 'page': page.number})
            chunks.append(chunk)
    # Tạo DB
    # Chroma.from_documents(chunks, model, persist_directory="./chroma_db")

    # print(chunks)
    return chunks

sample_pdf_path = ["storage/LNCT800SoftwareApplicationManual.pdf"]
all_chunks = []
all_docs = []

for path in sample_pdf_path:
    chunk = insert_pdf_to_db(path)
    all_chunks.extend(chunk)

import weaviate
from weaviate.gql.get import HybridFusion
client = weaviate.Client(
    url="http://localhost:8081"
)

db = Weaviate.from_documents(all_chunks, model, client = client, by_text=False)

# Perform similarity search

query = "INT3170"
vector = model.embed([query])[0]  # Get the embedding vector for the query
docs = db.similarity_search(vector, k=4)


# Print the results
print(docs)


response = (
    client.query
    .get("JeopardyQuestion", ["question", "answer"])
    .with_hybrid(query="INT3170", alpha=0.5)
    .with_limit(5)
    .do()
)

query_results = (
    client.query
    .get(
        class_name="Article",  # Replace with the actual class name in your schema
        properties=[
            "content",
            "source",
            "page",
        ],
    )
    .with_additional(properties=["score"])
    .with_autocut(2)
    .with_hybrid(
        query=query,
        fusion_type=HybridFusion.RELATIVE_SCORE,
        properties=[
            "content",
        ],
    )
    .do()
)


print(query_results)