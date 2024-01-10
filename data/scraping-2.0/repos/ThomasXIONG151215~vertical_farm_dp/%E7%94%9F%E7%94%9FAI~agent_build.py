from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory,ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from pypdf import PdfReader
from io import BytesIO
from typing import Any, Dict, List
import re
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain import (LLMChain, 
                       OpenAI,
                       SerpAPIWrapper,
                       LLMMathChain)
from langchain.agents import ZeroShotAgent,create_csv_agent
from langchain.memory import ConversationBufferMemory,ConversationEntityMemory
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentExecutor,create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import streamlit as st
from streamlit_chat import message
from analysis_tools import get_best_combination, analyze_growth_performance
from langchain.chat_models import ChatOpenAI
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.agents import AgentType
from langchain.agents import initialize_agent

api = "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"
serapi = "e49a7c10888561af955ab5fcf880d3612e7484e28454f43ad8cf5244e48aeda4"

def prompt_llm_chain_agent_executor():

    llm = ChatOpenAI(temperature=0,model="gpt-3.5-turbo-0613",openai_api_key=api)
    
    search = SerpAPIWrapper(serpapi_api_key=serapi)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    pd_agent = create_pandas_dataframe_agent(
    llm,
    st.session_state.df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS
    )
    tools = [
        Tool(
            name="Best Combination",
            func=get_best_combination,
            description="Analyse the newest data and describe potential best parameters that empower plant growth"
        ),
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
        ),
        #Tool(
        #    name="Pandas",
        #    func=pd_agent.run,
        #    description="useful for when you need to answer questions about dataframes",
        #)
    ]
    prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available. 
                        You have access to a single tool:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    from langchain.schema import messages_to_dict
    from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

    prompt = ENTITY_MEMORY_CONVERSATION_TEMPLATE

    
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain, ConversationChain
    import json
    from langchain import LLMChain

    if "data_memory" not in st.session_state:
        
        try:
            retrieved_db = open("chat_history.json", "r").read()
            retrieve_from_db = json.loads(retrieved_db)
            retrieved_messages = messages_from_dict(retrieve_from_db)
            retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
            retrieved_memory = ConversationEntityMemory(chat_memory=retrieved_chat_history, memory_key="chat_history")
            memory = retrieved_memory

        except:
            #个体记忆
            #初始化
            memory = ConversationEntityMemory(
                llm=llm,
                #memory_key="chat_history"
            ) 
            _input = {"input": "Deven & Sam are working on a hackathon project"}
            memory.load_memory_variables(_input)
            memory.save_context(
                _input,
                {"output": " That sounds like a great project! What kind of project are they working on?"}
            )
            
        
        st.session_state.data_memory = memory

    #llm = OpenAI(model="gpt-3.5-turbo-0613", openai_api_key=api)

    agent_executor = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
        verbose=True, 
        memory=st.session_state.data_memory)

    query = st.text_input(
        "**累计知识问答**",
        placeholder="输入",
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []#[{"role":"user","content": "你好，生生AI?"}]
        #st.session_state["messages"].append({"role":"bot","content": "你好，我是生生AI，我可以回答你的问题，你可以问我关于生物医学的任何问题。"})

    if query:
        #message(query, is_user=True)
        st.session_state["messages"] += [{"role":"user","content": query}]
        res = agent_executor.run(query)
        message(res, is_user=False)
        st.session_state["messages"] += [{"role":"bot","content": res}]
        #存储对话历史
        extracted_messages = st.session_state.data_memory.chat_memory.messages
        ingest_to_db = messages_to_dict(extracted_messages)
        json.dump(ingest_to_db, open("chat_history.json", "w"))


def pandas_agent():
    llm = ChatOpenAI(temperature=0,model="gpt-3.5-turbo-0613",openai_api_key=api)
    pd_agent = create_pandas_dataframe_agent(
        llm,
        st.session_state.df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS
        )
    
    query = st.text_input(
        "**累计数据问答**",
        placeholder="输入",
    )

    if "pd messages" not in st.session_state:
        st.session_state["pd messages"] = []#[{"role":"user","content": "你好，生生AI?"}]
        #st.session_state["messages"].append({"role":"bot","content": "你好，我是生生AI，我可以回答你的问题，你可以问我关于生物医学的任何问题。"})
    #展示整个对话历史
    #for msg in st.session_state["messages"]:
    #    message(msg["content"], msg["role"]) 

    if query:
        #message(query, is_user=True)
        st.session_state["pd messages"] += [{"role":"user","content": query}]
        res = pd_agent.run(query)
        message(res, is_user=False)
        st.session_state["pd messages"] += [{"role":"bot","content": res}]
    
    #st.write(pd_agent.run("the average temperature"))

# Define a function to parse a PDF file and extract its text content
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output

# Define a function to convert text content to a list of documents
def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks

def read_pdf_and_store_memory():

    # Allow the user to upload a PDF file
    uploaded_file = st.file_uploader("**请上传pdf材料**", type=["pdf"])
    st.write(uploaded_file)

    if uploaded_file:
        name_of_file = uploaded_file.name
        doc = parse_pdf(uploaded_file)
        pages = text_to_docs(doc)
        # Define a function for the embeddings
    
        if pages:
            # Allow the user to select a page and view its content
            with st.expander("Show Page Content", expanded=False):
                page_sel = st.number_input(
                    label="Select Page", min_value=1, max_value=len(pages), step=1
                )
                pages[page_sel - 1]
            # Allow the user to enter an OpenAI API key

            def test_embed():
                embeddings = OpenAIEmbeddings(openai_api_key=api)
                # Indexing
                # Save in a Vector DB
                with st.spinner("It's indexing..."):
                    index = FAISS.from_documents(pages, embeddings)
                st.success("Embeddings done.", icon="✅")
                return index

            #if api:
            # Test the embeddings and save the index in a vector database
            index = test_embed()
            # Set up the question-answering system
            qa = RetrievalQA.from_chain_type(
                llm=OpenAI(),
                chain_type="stuff",
                retriever=index.as_retriever(),
            )
            # Set up the conversational agent
            tools = [
                Tool(
                    name="State of Union QA System",
                    func=qa.run,#associating info read from the pdf
                    description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
                )
            ]
            prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available. 
                        You have access to a single tool:"""
            suffix = """Begin!"

            {chat_history}
            Question: {input}
            {agent_scratchpad}"""

            prompt = ZeroShotAgent.create_prompt(
                tools,
                prefix=prefix,
                suffix=suffix,
                input_variables=["input", "chat_history", "agent_scratchpad"],
            )

            if "memory" not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(
                    memory_key="chat_history"
                )

            llm_chain = LLMChain(
                llm=OpenAI(
                    temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo-0613"
                ),
                prompt=prompt,
            )
            agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
            agent_chain = AgentExecutor.from_agent_and_tools(
                agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
            )

            # Allow the user to enter a query and generate a response
            query = st.text_input(
                "**What's on your mind?**",
                placeholder="Ask me anything from {}".format(name_of_file),
            )

            if "pdf messages" not in st.session_state:
                st.session_state["pdf messages"] = []#[{"role":"bot","content": "Hello, how can I help you?"}]
            for msg in st.session_state["pdf messages"]:
                message(msg["content"], msg["role"])
            
            if query:
                #with st.spinner(
                #    "Generating Answer to your Query : `{}` ".format(query)
                #):
                message(query, is_user=True)
                st.session_state["pdf messages"] += [{"role":"user","content": query}]
                res = agent_chain.run(query)
                message(res, is_user=False)
                st.session_state["pdf messages"] += [{"role":"bot","content": res}]
                
                #    st.info(res, icon="🤖")

            # Allow the user to view the conversation history and other information stored in the agent's memory
            with st.expander("History/Memory"):
                st.session_state.memory

def csv_agent(csv_address):
    agent = create_csv_agent(
        OpenAI(temperature=0,openai_api_key=api, model="gpt-3.5-turbo-0613"),
        csv_address,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        )
    # Allow the user to enter a query and generate a response
    query = st.text_input(
        "**数据集问答**",
        placeholder="Ask me anything from {}".format(csv_address[:-3]),
    )
    if query:
        with st.spinner(
            "Generating Answer to your Query : `{}` ".format(query)
        ):
            res = agent.run(query)
            st.info(res, icon="🤖")


