from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document

# Our sentences we like to encode
sentences = [
    "tell me something about a brown fox",
    "This framework generates embeddings for each input sentence",
    "Sentences are passed as a list of string.",
    "The quick brown fox jumps over the lazy dog.",
]

# huggingface.DEFAULT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L12-v2'
# HuggingFaceEmbeddings.model_name = 'sentence-transformers/all-MiniLM-L12-v2'

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
db = None
docs = []
for sentence in sentences:
    doc = Document(page_content=sentence, metadata={"source": "local"})
    if db is None:
        db = FAISS.from_documents([doc], embeddings)
    else:
        db.add_documents([doc])
    # docs.append(doc)
    # FAISS.from_documents([doc],embeddings)

# db = FAISS.from_documents(docs, embeddings)
query = "tell me something about a brown fox"
docs_and_scores = db.similarity_search_with_score(query)
# print(docs_and_scores[0])
# print(docs_and_scores)
# print(embeddings.model_name)
parsed_results = []
for doc, score in docs_and_scores:
    parsed_result = {"content": doc.page_content, "score": score}
    parsed_results.append(parsed_result)

print(parsed_results)
