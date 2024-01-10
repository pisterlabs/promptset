#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
# from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import time
import streamlit as st
from ingest import main as ingest_main
from langchain.chat_models import ChatOpenAI
# from langchain.llms import OpenAI

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS

def main():
    if 'qa' not in st.session_state:
        print("No Model")
        st.session_state.qa = None

    qa = st.session_state.qa
    args = parse_arguments()
    # configure the streamlit app
    st.set_page_config(page_title="privateGPT", page_icon="💬", layout="wide")
    st.title("Private GPT: Ask questions to your documents.")
    # header of the page
    st.header("Chat with your documents")
    givenKey = False
    # add side bar
    with st.sidebar:
        st.subheader("OPENAI API KEY")
        st.write("Please enter your OPENAI API KEY")
        api_key = st.text_input("Enter your API KEY: ",key="api_key")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            givenKey = True

        if givenKey:
            st.subheader("Your Documents")
            files = st.file_uploader('Upload your documents', type=['pdf', 'txt', 'docx', 'doc', 'pptx', 'ppt', 'csv','enex','eml','epub','html','md','odt',], accept_multiple_files=True, key="upload")
            # save the uploaded files in source_documents folder
            if files:
                for file in files:
                    with open(os.path.join("source_documents", file.name), "wb") as f:
                        f.write(file.getbuffer())
        
            if st.button("Process Documents") and files:
                # add a spiner
                with st.spinner("Processing your documents..."):
                    # get the text in the documents
                    print("ingesttttttttt")
                    ingest_main()
                    print("after ingesttttttttt")
                    # Parse the command line arguments
                    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
                    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
                    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
                    # activate/deactivate the streaming StdOut callback for LLMs
                    # callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
                    # Prepare the LLM
                    # match model_type:
                    #     case "LlamaCpp":
                    #         llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
                    #     case "GPT4All":
                    #         llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
                    #     case _default:
                    #         # raise exception if model_type is not supported
                    #         raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
                    
                    llm = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
                    # llm = OpenAI(model_name='text-ada-001',temperature=0.7,openai_api_key=os.environ.get("OPENAI_API_KEY"))
                    # memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
                    # conversation_chain = ConversationalRetrievalChain.from_llm(
                    #     llm=llm,
                    #     retriever=retriever,
                    #     memory=memory
                    # )
                    # create the qa model with conversation_chain
                    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
                    # save the state of the qa variable
                    st.session_state.qa = qa
        else:
            # warning message
            st.warning("Please enter your API KEY first",icon="⚠")
            files = None
        
    print("take input")
    # create text input
    query = st.text_input("Enter a query: ",key="query")
    if query:
        # empty the text_input after pressing enter
        # st.session_state.query = ""
        # write the query to the screen
        st.write(f"\n## Question: {query}")
        # then pass the query to the qa model
        with st.spinner("Waiting for the response..."):
        # Get the answer from the chain
            if not givenKey:
                # error
                st.error("Please enter your API KEY first.",icon="⚠")
                return
            if st.session_state.qa is None:
                # error msg
                st.error("Please process your documents first.",icon="⚠")
                return
            start = time.time()
            # print("query", query)
            res = qa(query)
            print("res", res)
            # res = "This is a test"
            answer, docs = res['result'], [] if args.hide_source else res['source_documents']
            end = time.time()
            # write the answer to the screen
            st.write(f"## Answer:")
            st.write(f"> {answer}")
            # Print the time taken to answer the question
            # Print the relevant sources used for the answer
            st.write(f"> Answer (took {round(end - start, 2)} s.):")
            st.write(f"### Relevant sources:")
            for i in range(len(docs)):
                # print(docs[i].metadata)
                st.write(f"##### Souce no. {i+1} :" + docs[i].metadata["source"] + ":")
                try:
                    st.write(f'>>> *Page no. {docs[i].metadata["page"]}* : '+docs[i].page_content)
                except:
                    st.write(f'>>>'+docs[i].page_content)     

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
