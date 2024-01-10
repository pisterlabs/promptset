import os
import streamlit as st
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent
from langchain.memory import StreamlitChatMessageHistory, ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate

from langchain import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Fixed this line

st.set_page_config(layout="wide", page_icon="💬", page_title="HKSTP🤖")

st.markdown(
            f"""
            <h1 style='text-align: center;'> I'am your agent to answer quetions related to HKSTP 😁</h1>
            """,
            unsafe_allow_html=True,
        )


# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Initialize index and retriever
embeddings = OpenAIEmbeddings()


index_name_ideation = "hkstp-ideation"
index_name_incubation = "hkstp-incubation"
index_name_acceleration = "hkstp-accleration"
index_name_elite = "hkstp-elite"
index_name_funds = "hkstp-funds"
index_name_infrastruture = "hkstp-infrastruture"


ideation_search = Pinecone.from_existing_index(index_name_ideation , embeddings)
incubation_search = Pinecone.from_existing_index(index_name_incubation , embeddings)
acceleration_search = Pinecone.from_existing_index(index_name_acceleration , embeddings)
elite_search = Pinecone.from_existing_index(index_name_elite , embeddings)
funds_search = Pinecone.from_existing_index(index_name_funds , embeddings)
infrastruture_search = Pinecone.from_existing_index(index_name_infrastruture , embeddings)


retriever_ideation =ideation_search.as_retriever(search_kwargs={"k":10})
retriever_incubation =incubation_search.as_retriever(search_kwargs={"k":10})
retriever_acceleration =acceleration_search.as_retriever(search_kwargs={"k":10})
retriever_elite =elite_search.as_retriever(search_kwargs={"k":10})
retriever_funds =funds_search.as_retriever(search_kwargs={"k":10})
retriever_infrastrature =infrastruture_search.as_retriever(search_kwargs={"k":10})


# Initialize GPT-4 for chat
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4',
    temperature=0.5
)



# Initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=7,
    return_messages=True
)


prompt = PromptTemplate(
    input_variables = ["query"],
    template = "{query}"
)

llm_chain = LLMChain(llm = llm, prompt = prompt)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)


# Initialize QA tool
# retrieval qa chain

ideation_gpt = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever_ideation
)

# retrieval qa chain
incubation_gpt = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever_incubation
)

# retrieval qa chain
acceleration_gpt = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever_acceleration
)

# retrieval qa chain
elite_gpt = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever_elite
)

# retrieval qa chain
funds_gpt = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever_funds
)


# retrieval qa chain
infrastrature_gpt = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever_infrastrature
)


tools = [
    
    Tool(
        name ='Language Model',
        func = llm_chain.run,
        description =(
             'use this tool for general purpose queries and logic'
        )
    ),
    
    Tool(
        name='incubation_gpt',
        func=incubation_gpt.run,
        description=(
            'Useful when answering knowledge queries about the hkstp incubation and incubio program'
        )
    ),
      
    Tool(
        name='ideation_gpt',
        func=ideation_gpt.run,
        description=(
            'Useful when answering queries about the hkstp ideation program'
        )
    ),
    
     Tool(
        name='acceleration_gpt',
        func=acceleration_gpt.run,
        description=(
            'Useful when answering queries about the hkstp acceleration program'
        )
    ),
    
    Tool(
        name='elite_gpt',
        func=elite_gpt.run,
        description=(
            'Useful when answering knowledge queries about the hkstp elite program'
        )
    ),
    
   Tool(
        name='funds_gpt',
        func=funds_gpt.run,
        description=(
            'Useful when answering knowledge queries about the hk government funds for i&t'
        )
    ), 
    
    Tool(
        name='infrastrature_gpt',
        func=infrastrature_gpt.run,
        description=(
            'Useful when answering knowledge queries about hkstp ITR and INFRASTRUTURE'
        )
    )
    
]
    
  
# Initialize Streamlit components
st_callback = StreamlitCallbackHandler(st.container())


# Initialize the agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    max_iterations=7,
    verbose=True,
    memory=memory)


# If no messages exist, start the conversation
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

# Display past chat messages
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Capture and process new user input


if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)

