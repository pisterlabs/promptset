# load_n_split.py
# dH 12/5/23
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import time

FILE_PATH = "C:\\2023_fall\\aiChatBotDev\\pdfs\\howTheArmyRuns.pdf"

# Create PDF loader
loader = PyPDFLoader(FILE_PATH)

# split the document into pages
pages = loader.load_and_split()
print("\n pages created:")
print(len(pages))


# create the vectors
# all-MiniLM-L6-v2 is a a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector
# space and can be used for tasks like clustering or semantic search.
#
# Usage (Sentence-Transformers)
# Using this model becomes easy when you have sentence-transformers installed:
#
# pip install -U sentence-transformers
#
# Then you can use the model like this:from sentence_transformers import SentenceTransformer
# sentences = ["This is an example sentence", "Each sentence is converted"]
#
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings = model.encode(sentences)
# print(embeddings)

embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Start timer
start_time = time.time()

# Create vector store with progress bar.
# use Pinecone or Weaviate in the future, for
# now this is Chroma
vectordb = Chroma.from_documents(
    documents=pages,
    embedding=embedding_function,
    persist_directory="../vector_db",
    collection_name="the_army_way"
)

# Stop timer
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Embedding process completed in {elapsed_time:.2f} seconds.")


# make persistent
vectordb.persist()



