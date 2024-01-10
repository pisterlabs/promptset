# document_qa.py

# Import required libraries for document QA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.llms import OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone
import openai
# (Keep the rest of the imports)
# Add any imports that you need for your specific implementation

def get_document_answer(user_question):
# Put the document QA code here (exclude the imports)
    OPENAI_API_KEY = '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'
    PINECONE_API_KEY = '9dde6c4d-1d67-421c-b5d2-babfc9f67c2c'
    PINECONE_API_ENV = 'us-west4-gcp'
    embeddings= OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )
    index_name = "langchain2"
    docsearch = Pinecone.from_existing_index(index_name="langchain2", embedding=embeddings)
    query = "What is environmental determinism"
    docs = docsearch.similarity_search(query)
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY,model="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")
    # query is user input which needs to be brought in
    docs = docsearch.similarity_search(user_question)
    answer = chain.run(input_documents=docs, question=user_question)
    return answer
