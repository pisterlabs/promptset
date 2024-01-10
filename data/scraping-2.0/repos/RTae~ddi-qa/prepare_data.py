import streamlit as st
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, AgentType
from langchain.vectorstores import ElasticVectorSearch
from langchain.embeddings import VertexAIEmbeddings
from langchain.retrievers import PubMedRetriever
from langchain.agents import Tool, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from google.cloud import aiplatform
from langchain.llms import VertexAI
import os

def setup_interface(sti: st):
    sti.set_page_config(
        page_icon=":robot:",
        page_title="DDI QA"
    )
    sti.title("💊 DDI QA")

    if "messages" not in sti.session_state:
        sti.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, I'm a chatbot who can help with your research."}
        ]

    for msg in sti.session_state.messages:
        sti.chat_message(msg["role"]).write(msg["content"])
    
def handle_prompt_interface(agent: AgentExecutor,sti: st):
    if prompt := sti.chat_input(placeholder="Put your question here"):
        sti.session_state.messages.append({"role": "user", "content": prompt})
        sti.chat_message("user").write(prompt)
        answer = agent(prompt)
        print(answer)
        answer = answer['output']

        with sti.chat_message("assistant"):
            sti.session_state.messages.append({"role": "assistant", "content": answer})
            sti.write(answer)


def setup_llm():
    PROJECT_ID=os.environ.get("PROJECT_ID")
    REGION=os.environ.get("REGION", "asia-southeast1")
    ES_URL=os.environ.get("ES_URL")
    ES_INDEX_NAME="data-index"

    if(PROJECT_ID is None or ES_URL is None):
        raise SystemError("Please set PROJECT_ID and ES_URL environment variable")

    aiplatform.init(project=f"{PROJECT_ID}", location=f"{REGION}")
    embeddings_service = VertexAIEmbeddings()

    db = ElasticVectorSearch(ES_URL, ES_INDEX_NAME, embedding=embeddings_service)
    prompt_template = """
        Now you are drug researcher who could answer interaction between drug and it type of interaction and I give you type of interaction, between drug and reseach sentence
        by using the following pieces of context to answer and data I gave you.
        {context}
        {question}    
        Answer list of drug might interact with:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    llm = VertexAI(temperature=0.2)
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )
    qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=False
        )

    qa_pubmed = PubMedRetriever()

    tools = [
        Tool(
            name='Drug Interaction Question Answering',
            func=qa.run,
            description=(
                'This tool allows you to ask questions about drug interaction and get answers. '
            )
        ),
        Tool(
            name='Pubmed',
            func=qa_pubmed.run,
            description=(
                'PubMedRetriever is a tool that allows you to search for articles in the PubMed database.'
            )
        ),
    ]

    agent = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=conversational_memory
    )

    return agent

def main():
    agent = setup_llm()
    setup_interface(st)
    handle_prompt_interface(agent, st)

if __name__ == "__main__":
    main()