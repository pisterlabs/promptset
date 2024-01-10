import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import DeepLake
from langchain.agents import AgentType, Tool, initialize_agent

# Load environment variables
load_dotenv()

# Initialize the question-answering system
def setup_qa_system():
    # Embeddings model setup
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Instantiate the LLM model
    llm = OpenAI(model="text-davinci-003", temperature=0)

    # Create Deep Lake dataset
    my_activeloop_org_id = "mpazaryna"
    my_activeloop_dataset_name = "langchain_course_from_zero_to_hero"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
    db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

    # Setup Retrieval QA
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=db.as_retriever()
    )

    # Initialize agent with tools
    tools = [
        Tool(
            name="Retrieval QA System",
            func=retrieval_qa.run,
            description="Useful for answering questions.",
        ),
    ]

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )

    return agent

# Initialize the agent
agent = setup_qa_system()

# Streamlit app starts here
st.title('Question Answering System')

# User input
user_question = st.text_input('Ask a question:', '')

if st.button('Get Answer'):
    if user_question:
        try:
            response = agent.run(user_question)
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question to get an answer.")

