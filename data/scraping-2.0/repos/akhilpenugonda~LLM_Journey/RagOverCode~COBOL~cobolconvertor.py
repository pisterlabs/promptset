from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatGooglePalm
from langchain.chains import RetrievalQA
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
import google.generativeai as palm



repo_path = "/Users/akhilkumarp/development/personal/github/LLM_Journey/RagOverCode/COBOL/game"
loader = GenericLoader.from_filesystem(
    repo_path,
    glob="**/*.cbl",
    suffixes=[".cbl"],
    parser=LanguageParser(language=Language.COBOL, parser_threshold=500),
)
documents = loader.load()

# palm.configure(api_key='key')
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(texts, embeddings)

query = "What are the different types of concepts explained"
matching_docs = db.similarity_search(query)

matching_docs[0]

persist_directory = "chroma_db"

vectordb = Chroma.from_documents(
    documents=texts, embedding=embeddings, persist_directory=persist_directory
)

vectordb.persist()
import os
os.environ["OPENAI_API_KEY"] = "key"

model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)
# llm = ChatGooglePalm(model_name="text-bison-001", google_api_key="key")
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

query = "Generate the complete code in a single java file, with all the dependencies"
matching_docs = db.similarity_search(query)
answer =  chain.run(input_documents=matching_docs, question=query)
print(answer)

