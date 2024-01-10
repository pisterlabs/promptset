
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import SVMRetriever, TFIDFRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate




import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']



# Change the path to the PDF you want to load
loader = PyPDFLoader("docs/The 17 immutable laws In Implant Dentistry.pdf")
pages = loader.load()

len(pages)

# Custom function to include metadata during document splitting
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1500,
    chunk_overlap=150,
    length_function=len
)

def split_documents_with_metadata(pages):
    docs_with_metadata = []
    for page in pages:
        page_content = page.page_content
        page_splits = text_splitter.split_text(page_content)
        for split in page_splits:
            doc_with_metadata = {
                'text': split,
                'metadata': {
                    'page': page.metadata['page_number'],  # get the page number from the metadata
                }
            }
            docs_with_metadata.append(doc_with_metadata)
    return docs_with_metadata

docs = split_documents_with_metadata(pages)
splits = text_splitter.split_documents(docs)

print(len(docs))



# Specify your persist directory
persist_directory = 'docs/chroma/'

# Create an instance of OpenAIEmbeddings
embedding = OpenAIEmbeddings()

# Create an instance of Chroma using the from_documents class method
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory,
)



# Print vectordb to verify
print(vectordb)
print("The size of the vectorstore is: ", vectordb._collection.count())

llm = ChatOpenAI(model_name="gpt-4", temperature=0)



len(docs)

print(len(docs))


# Build prompt
template = """ You are a very important professor at a dental school. You are answering questions from your students about the course material.
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)



# Memory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

question = "What relevance do provisionals have in implant dentistry?"

result = qa_chain({"query": question})

result["result"]

print(result["result"])
for doc in result['source_documents']:
    print(doc.metadata)

question = "What is the key about socket shield technique?"

result = qa_chain({"query": question})

result["result"]

print(result["result"])
for doc in result['source_documents']:
    print(doc.metadata)