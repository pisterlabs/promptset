# from langchain.llms import LlamaCpp
# from langchain import PromptTemplate, LLMChain
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# template = """Question: {question}

# Answer: Let's think step by step."""

# prompt = PromptTemplate(template=template, input_variables=["question"])

# # Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# # Verbose is required to pass to the callback manager

# # Make sure the model path is correct for your system!
# llm = LlamaCpp(
#     model_path="./ggml-model-q4_0.bin", callback_manager=callback_manager, verbose=True
# )

# llm_chain = LLMChain(prompt=prompt, llm=llm)

# question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

# llm_chain.run(question)

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.chains import RetrievalQA
import pickle
import os
from streamlit_chat import message
from dotenv import load_dotenv
from raw_strings import *
import openai

st.set_page_config(page_title="🏡 LeaseGPT", page_icon=":door:")


def get_listings_tool(retriever):
    tool_desc = '''Use this tool to inform user about listings from context. Give the user 2 options based on their criterion. If the user asks a question that is not in the listings, the tool will use OpenAI to generate a response.
    This tool can also be used for follow up quesitons from the user. 
    '''
    tool = Tool(
        func=retriever.run,
        description=tool_desc,
        name="Lease Listings Tool",   
    )
    return tool


def get_text_chunks(selection: str):
    # TODO: Scraping Craigslist
    text = " ".join([doc1, doc2, doc3, doc4])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50, length_function=len
    )
    docs = [doc1, doc2, doc3, doc4]
    chunks = []

    # Splitting the text into chunks
    for doc in docs:
        if len(doc) > 1200:
            chunk_doc = text_splitter.split_text(doc)
            for chunk in chunk_doc:
                chunks.append(chunk)
        else:
            chunks.append(doc)
    return chunks


def get_set_vector_store(chunks, selection):
    embeddings = OpenAIEmbeddings()
    store_name = "craigslist_vector_store"
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            vector_store = pickle.load(f)
            st.write("Embeddings Loaded from the Disk")
    else:
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vector_store, f)
            st.write("Embeddings Created and Saved to Disk")
    return vector_store


def setup_leasing_agent(vector_store, api_key):
    # Template for the chatbot
    template = """I want you to act to act like a leasing agent for me. Giving me the best options always based on what you read below. 
        You can give me something which matches my criteria or something which is close to it.
        """
    
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0, model_name="gpt-3.5-turbo")

    retriever = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vector_store.as_retriever()
    )

    tools = [get_listings_tool(retriever=retriever)]
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=3, return_messages=True
    )

    conversational_agent = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=2,
        early_stopping_method="generate",
        memory=memory,
    )

    conversational_prompt = conversational_agent.agent.create_prompt(
        system_message=template,
        tools=tools,
    )

    conversational_agent.agent.llm_chain.prompt = conversational_prompt
    print('Prompt', conversational_prompt)
    return conversational_agent


def main():
    os.environ["OPENAI_API_KEY"] = ""
   
    # api_key = st.text_input("Please enter your OpenAI key")
    api_key = ""
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    load_dotenv()

    if os.environ["OPENAI_API_KEY"] is not None:
        print("OPEN AI Key", os.environ["OPENAI_API_KEY"])
        selection = ""
        chunks = get_text_chunks(selection)

        # print("Prompt", prompt)
        try:
            vectore_store = get_set_vector_store(chunks, selection)
            query = "Show me houses near UW"

            if query:
                # docs = vectore_store.similarity_search(query, k=3)
                leasing_gpt = setup_leasing_agent(vectore_store, api_key)
                print("Leasing GPT", leasing_gpt.run(query))
                # with get_openai_callback() as callback:
                    # st.write(conversational_agent("Give a few houses near UW"))
                    # print("Output", op)
                    # st.write(op)

        except openai.error.AuthenticationError as e:
            print("Error", e)
            
        except:
            if os.environ["OPENAI_API_KEY"] is None:
                # 
                print("No API key found")
            # if e == "No API key found":


if __name__ == "__main__":
    main()
