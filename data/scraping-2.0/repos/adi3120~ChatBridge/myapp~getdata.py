from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

# load the document and split it into chunks
loader = TextLoader("./data.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# # create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function)


db2 = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
# query it
# query = "just give the names of the companies and nothing else"
# docs = db2.similarity_search(query)


# db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
# docs = db3.similarity_search("which companies come for placements")
# print(docs[0])
