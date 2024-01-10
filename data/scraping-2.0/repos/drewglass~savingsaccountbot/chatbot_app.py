import os
import pinecone
from langchain import OpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

# load the documents and split them. 
# This can be substituted with any data type loader according to your requirement.
loader = TextLoader('example_data/savings-accounts.txt')
documents = loader.load()

# We proceed by segmenting the documents and generating their embeddings. 
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

# initialize pinecone
# get the api key and environment from the .env file
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENV')
)

# create an index in pinecone and get the name 
# https://docs.pinecone.io/docs/manage-indexes#create-an-index-from-a-public-collection
index_name = "langchain-demo"

# Store the documents and embeddings in the pinecone vectorstore. 
# This setup facilitates semantic searching throughout the documents.
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

# initialize the LLM
llm = OpenAI(temperature=0)
# the non-streaming LLM for questions
question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
# ConversationalRetrievalChain with a streaming llm for the docs
streaming_llm = OpenAI(
    streaming=True, 
    callback_manager=CallbackManager([
        StreamingStdOutCallbackHandler()
    ]), 
    verbose=True,
    temperature=0
)
doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=QA_PROMPT)

# initialize ConversationalRetrievalChain chabot
qa = ConversationalRetrievalChain(
    retriever=docsearch.as_retriever(), combine_docs_chain=doc_chain, question_generator=question_generator)

chat_history = []
question = input("Hi! Ask me a question about savings accounts. ")

# create a loop to ask the chatbot questions 
while True:
    result = qa(
        {"question": question, "chat_history": chat_history}
    )
    #result = qa({"question": question})
    print("\n")
    chat_history.append((result["question"], result["answer"]))
    question = input()