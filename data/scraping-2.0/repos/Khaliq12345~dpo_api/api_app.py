from fastapi import FastAPI
import model
import pickle
import fa_config
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import pickle
from langchain.schema import messages_from_dict, messages_to_dict
import json
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
import random, string

def save_chain(chain, filename):
    extracted_messages = chain.memory.chat_memory.messages
    ingested_db = messages_to_dict(extracted_messages)
    with open(filename, 'w') as f:
        json.dump(ingested_db, f)
        
def load_chain(filename):
    #load the model
    with open(filename, 'r') as f:
        loaded_db = json.load(f)
    retrieved_messages = messages_from_dict(loaded_db)
    retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
    retrieved_memory = ConversationBufferMemory(
        chat_memory=retrieved_chat_history, memory_key='chat_history', return_messages=True)
    vectorstore = model.get_vector_store()
    new_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(openai_api_key=fa_config.openai_key), retriever=vectorstore, memory=retrieved_memory
    )
    return new_chain

def save_or_renew_chain(conversation_chain):
    pickled_str = pickle.dumps(conversation_chain.memory)
    if len(conversation_chain.memory.chat_memory.messages) == 20:
        new_conversation_chain = model.get_chain()
        return new_conversation_chain
    else:
        vectorstore = model.get_vector_store()
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(openai_api_key=fa_config.openai_key), 
            retriever=vectorstore.as_retriever(), 
            memory=pickle.loads(pickled_str),
        )
        return conversation_chain  

def handle_userinput(user_question, conversation_chain):
    response = conversation_chain({'question': user_question})
    return response['answer']

def randomword():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(10))
    

app = FastAPI()

@app.get("/")
def root(query: str):
    conversation_chain = model.get_chain()
    answer = {
        "question": f"{query}",
        'answer': handle_userinput(query, conversation_chain)
        }
    return answer