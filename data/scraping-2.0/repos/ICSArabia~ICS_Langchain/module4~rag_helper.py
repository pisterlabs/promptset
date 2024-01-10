from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from langchain.tools import DuckDuckGoSearchRun
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import ast


def qa_chain_with_memory_and_search(retriever):

    # create the model
    llm = ChatOpenAI(model='gpt-4',
                        temperature=0.8)


    # create the chain to answer questions 

    global qa_chain
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
                                      chain_type="stuff",
                                      retriever=retriever,
                                    #   return_source_documents=True   #<---- for me to test / see all output on term
                                    )

    # memory buffer
    memory = ConversationBufferMemory(memory_key="chat_history", k=5, return_messages=True)

    # internet search
    search = DuckDuckGoSearchRun()
    
    return llm,qa_chain,memory,search

def ask_with_memory(vector_store, question, chat_history=[]):

    llm = ChatOpenAI(temperature=0.7)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":5})
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    
    return result, chat_history

def break_response_source(response):
    try:
        dictionary = ast.literal_eval(response)
        return (dictionary['answer'], dictionary['sources'])
    except Exception:
        source = 'source missing not asked'
        return response,source