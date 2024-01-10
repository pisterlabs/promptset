from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
 

# Creating an UnstructuredFileLoader to load text.
loader = DirectoryLoader('Documents')

# Loading raw documents using document_loaders from LangChain
raw_documents = loader.load()
print("Document loaded using document_loaders from LangChain")

# Initializing a CharacterTextSplitter for text splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100,
)

# Splitting text into documents using text_splitter from LangChain
documents = text_splitter.split_documents(raw_documents)
print(documents)
print("Text splitted using text_splitter from LangChain")

# Initializing OpenAIEmbeddings for text embeddings
embeddings = OpenAIEmbeddings()

# Creating a vectorstore using FAISS from LangChain
vectorstore = FAISS.from_documents(documents, embeddings)
print("created a vectorestore using FAISS from LangChain")


# Saving the vectorstore locally
vectorstore.save_local("faiss_index_constitution")
print("Vectorestore saved in 'faiss_index_constitution' ")

