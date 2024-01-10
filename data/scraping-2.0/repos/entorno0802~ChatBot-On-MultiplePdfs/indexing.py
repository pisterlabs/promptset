from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()
loader = TextLoader('state_of_the_union.txt', encoding='utf8')
index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma, embedding=OpenAIEmbeddings(), text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders([loader])
query = "what did the president say about education?"
print(index.query(query))
print(index.query_with_sources(query))
