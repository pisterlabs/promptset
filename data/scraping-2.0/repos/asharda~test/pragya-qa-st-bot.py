import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
import pinecone
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)
index = pinecone.Index(index_name=os.environ["PINECONE_INDEX_NAME"])

# Initialize ConversationBufferMemory
memory_buffer = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize PromptTemplate
prompt = PromptTemplate(
    input_variables=["query"],
    template="{query}"
)

# Initialize ChatOpenAI
llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize tools
search = DuckDuckGoSearchRun()
wolfram = WolframAlphaAPIWrapper()

# Initialize LLMChain and load QA chain
llm_chain = LLMChain(llm=llm, prompt=prompt)
chain = load_qa_chain(llm, chain_type="stuff")

# Initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

# Initialize Pinecone index for vector search
docsearch = Pinecone.from_existing_index(INDEX_NAME, embeddings)

def pinecone_query(query):
    docs = docsearch.similarity_search(query)
    answer = chain.run(input_documents=docs, question=query)
    print("Pinecone search completed")
    return answer

# Initialize tools using Tool class
llm_tool = Tool(name='Language Model', func=llm_chain.run, description="General purpose queries and logic")
search_tool = Tool(name='CurrentEvents', func=search.run, description="Answer questions about current events.")
wolfram_tool = Tool(name='Math-Science', func=wolfram.run, description="Answer questions about mathematics and science.")
pinecone_tool = Tool(name='Pinecone', func=pinecone_query, description="Answer questions about private content from the index.")

my_tools = [pinecone_tool, wolfram_tool, search_tool, llm_tool]

# Initialize agent using initialize_agent function
agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=my_tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    memory=memory_buffer,
)

def clear_query_and_answer():
    st.session_state.query = ''

if __name__ == "__main__":
    st.image('pragyalogo.png')
    st.subheader('Ask me about PIMA, Teachus, Pragya, or Math content:')

    query = st.text_input('Ask a question:', key='query')

    if query:
        if query.lower() in ["exit", "quit"]:
            sys.exit()

        answer = agent.run(input=query)

        st.text_area('Answer:', value=answer, key='answer', height=100)
        st.divider()

        if 'history' not in st.session_state:
            st.session_state.history = ''

        value = f'Q: {query}\nA: {answer}'
        st.session_state.history = f'{value}\n{"-" * 100}\n{st.session_state.history}'
        h = st.session_state.history

        st.text_area(label='Chat History', value=h, key='history', height=400)