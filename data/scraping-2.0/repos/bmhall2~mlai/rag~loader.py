from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

open_ai_key = ''

client = Chroma(
    collection_name="rag",
    persist_directory="./chroma"
)

client.delete_collection()

loader = DirectoryLoader(
    '/mnt/c/Users/Ben Hall/code/mlai/PdfTextTest/text/',
    glob="**/*.txt",
    # show_progress=True,
    loader_cls=TextLoader
)

docs = loader.load()

for x in range(len(docs)):
    docs[x].metadata['fileId'] = x

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

store = Chroma.from_documents(documents=all_splits,
                              embedding=OpenAIEmbeddings(api_key=open_ai_key),
                              collection_name="rag",
                              persist_directory="./chroma")
