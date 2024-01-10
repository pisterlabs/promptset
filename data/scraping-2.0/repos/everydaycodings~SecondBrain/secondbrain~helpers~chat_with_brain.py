from helpers.utils import load_embedding_model
from langchain.vectorstores import Chroma
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.utilities import SerpAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.chains import RetrievalQA
import streamlit as st


st.cache_resource
def load_db(model_name, device, persist_directory, search_kwargs):

        embedding = load_embedding_model(model_name, device)

        vectordb = Chroma(persist_directory="SecondBrain/secondbrain/database/{}".format(persist_directory), 
                  embedding_function=embedding)
        

        return vectordb
    

@st.cache_resource
def load_model(db_model, device, persist_directory, search_kwargs, model_architecture, model_name, model_path, max_token, temp, top_p, top_k):
    
    local_path = '{}/{}'.format(model_path, model_name)  # replace with your desired local file path
    callbacks = CallbackManager([StreamingStdOutCallbackHandler()])
    
    db = load_db(model_name=db_model, device=device, persist_directory=persist_directory, search_kwargs=search_kwargs)

    if model_architecture == "GPT4ALL":
        model = GPT4All(model=local_path, callbacks=callbacks, verbose=True, n_predict=max_token, temp=temp, top_p=top_p, top_k=top_k)
    if model_architecture == "Llama-cpp":
        model = LlamaCpp(model_path=local_path, callback_manager=callbacks, verbose=True, max_tokens=max_token,temperature=temp,top_p=top_p,top_k=top_k)
    
    qa = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    verbose=False,
    )

    return qa


def run_model(db_model, device, persist_directory, search_kwargs, model_architecture, model_name, model_path, max_token, temp, top_p, top_k, prompt):
     
    try:
        qa = load_model(db_model, device, persist_directory, search_kwargs, model_architecture, model_name, model_path[0], max_token, temp, top_p, top_k)
    except:
        qa = load_model(db_model, device, persist_directory, search_kwargs, model_architecture, model_name, model_path[1], max_token, temp, top_p, top_k)

    res = qa(prompt)

    return res["result"]