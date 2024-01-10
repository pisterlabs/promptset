# max_marginal_relevance.py
# dH 12/5/23
#  Fresno, CA
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

TEXT = ["Python is a high-level, interpreted programming language known for its clear syntax and readability, "
        "making it particularly suitable for beginners",
        "It supports multiple programming paradigms, including procedural, object-oriented, and functional "
        "programming, allowing for versatile application development.",
        "Widely used in various fields such as web development, data analysis, artificial intelligence, "
        "and scientific computing, Python has a vast ecosystem of libraries and frameworks.",
        "Additionally, its strong community support and open-source nature contribute to its continuous evolution and "
        "widespread adoption in the tech industry.",
        "Python is also renowned for its efficient code execution and dynamic typing system, which accelerates the "
        "development process and simplifies code maintenance, making it a preferred choice for rapid application "
        "development."]

meta_data = [{"source": "document 001", "page": 1},
             {"source": "document 002", "page": 2},
             {"source": "document 003", "page": 3},
             {"source": "document 004", "page": 4},
             {"source": "document 005", "page": 5}]

embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# this vector database is temporary, it is not persistent.
vector_db = Chroma.from_texts(
    texts=TEXT,
    embedding=embedding_function,
    metadatas=meta_data
)

response = vector_db.max_marginal_relevance_search(
    query="Tell me about a programming language used for data science", k=2
)

print(response)

metadatas=meta_data