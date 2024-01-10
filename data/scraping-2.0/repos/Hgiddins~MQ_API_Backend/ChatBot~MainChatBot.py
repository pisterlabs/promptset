import os
import threading
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.document_loaders import PyPDFLoader

from ChatBot.ChatBotConstants import APIKEY


# threadsafe chatbot to make logging in and out secure with chatbot data
class ThreadSafeChatbot:
    def __init__(self):
        self.retrieval_chain = None
        self.conversation_chain = None
        self.lock = threading.Lock()

    def boot(self):
        with self.lock:
            self.retrieval_chain, self.conversation_chain = boot_chatbot()

    def get_response(self, chain_type, question):
        with self.lock:
            if chain_type == "systemMessage":
                return get_issue_message_chatbot_response(self.retrieval_chain, self.conversation_chain, question)
            elif chain_type == "userMessage":
                return get_general_chatbot_response(self.retrieval_chain, self.conversation_chain, question)
            else:
                return "Invalid indicator value."


# utility functions
def setup_openai_authorization():
    """
    Sets up OpenAI's API authorization using environment variables or constants.
    """
    os.environ["OPENAI_API_KEY"] = APIKEY

def get_index(persist, loader):
    """
    Get or create the vector store index.

    Args:
        persist (bool): Whether to persist the vector store on disk.
        loader (object): Data loader.

    Returns:
        object: Initialized index.
    """
    if persist and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        return VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        if persist:
            return VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
        else:
            return VectorstoreIndexCreator().from_loaders([loader])


def instantiate_retrieval_chain(index):
    """
    Instantiates the conversational retrieval chain.
    Args:
        index (object): Vector store index.
    Returns:
        object: Initialized conversational retrieval chain.
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )
    return chain

def instantiate_conversation_chain():
    """
    Instantiates the conversation chain and its memory.
    Returns:
        tuple: Initialized memory and conversation chain objects.
    """
    memory = ConversationBufferWindowMemory( k=3, return_messages=True)
    conversation = ConversationChain(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        memory=memory
    )
    return memory, conversation


def boot_chatbot():
    PERSIST = False
    setup_openai_authorization()
    # loader = TextLoader("ChatBot/MQ_DOCS_2035.txt")
    loader = PyPDFLoader("ChatBot/mqIssuesDoc.pdf")
    index = get_index(PERSIST, loader)
    retrieval_chain = instantiate_retrieval_chain(index)
    memory, conversation_chain = instantiate_conversation_chain()


    return retrieval_chain, conversation_chain


def get_issue_message_chatbot_response(retrieval_chain, conversation_chain, issue_information):
    try:
        # get relevant related information from the documentation
        context_prompt = """[Context: IBM MQ] \n\n System Prompt: \nProvide relevant information about the causes and 
        possible solutions to this issue (include IBM Documentation hyperlinks where possible): """ + issue_information

        documentation_context = retrieval_chain({"question": context_prompt})['answer']

        troubleshoot_prompt = """System Prompt: \nYou are a helpful IBM MQ AI assistant. Given the context provided 
        give an overview of the problem in my system and how to fix it (include IBM Documentation hyperlinks where 
        possible). Think step by step. 
        \nSystem Issue Message:\n""" + issue_information + "\n\nIBMMQ Documentation Reference:\n"+documentation_context

        result = conversation_chain.predict(input=troubleshoot_prompt)

        return result

    except Exception as e:
        # Capture and return the error message
        return str(e)


def get_general_chatbot_response(retrieval_chain, conversation_chain, user_query):
    try:
        # get relevant related information from the documentation
        context_prompt = """[Context: IBM MQ] \n\n System Prompt: \nIf the user's query relates to 'IBM MQ', answer 
        informatively. For unrelated queries, reply with 'No information'. Think step by step.
        \nUser Prompt:\n""" + user_query

        documentation_context = retrieval_chain({"question": context_prompt})['answer']

        troubleshoot_prompt = """System Prompt: \nYou are a helpful IBM MQ AI assistant. If the user's query relates to 
        'IBM MQ', use the IBM MQ Documentation to answer informatively. For queries not about IBM MQ or general 
        pleasantries, reply that the question does not relate to IBM MQ and so you cannot answer it. Think step by step.
        \nUser Prompt:\n""" + user_query + "\n\nIBM MQ Documentation Reference:\n" + documentation_context

        result = conversation_chain.predict(input=troubleshoot_prompt)

        return result

    except Exception as e:
        # Capture and return the error message
        return str(e)


