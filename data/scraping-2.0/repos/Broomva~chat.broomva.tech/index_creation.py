# %%
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# %%
loader = DirectoryLoader(
    "../../../docs", glob="./*.md", loader_cls=TextLoader, recursive=True
)

documents = loader.load()
# %%

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# %%
embedding = OpenAIEmbeddings()

vectordb = FAISS.from_documents(documents=texts, embedding=embedding)

# %%
vectordb.save_local("docs.faiss")

# %%
