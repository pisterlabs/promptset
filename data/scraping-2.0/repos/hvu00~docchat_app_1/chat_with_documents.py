import logging
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


MODEL_NAME = "gpt-3.5-turbo"
INITIAL_CHAT_HISTORY = [{"role": "assistant", "content": "Train me with documents and I will use them in our chat!"}]


# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    _, extension = os.path.splitext(file)

    logging.info(f'Loading {file}')
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        logging.error('Document format is not supported!')
        return None

    data = loader.load()
    return data


# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answer(model_name, vector_store, question, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    from openai import ChatCompletion
    
    llm = ChatOpenAI(model=model_name, temperature=1)

    if vector_store:
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        answer = chain.run(question)
    else:
        answer = ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": question}
            ]
        )
        answer = answer.choices[0].message.content

    return answer


# calculate embedding cost using tiktoken
def calculate_embedding_cost(model_name, texts):
    import tiktoken
    enc = tiktoken.encoding_for_model(model_name)
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    logging.debug(f'Total Tokens: {total_tokens}')
    logging.debug(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004


def send_chat_msg():    
    chat_input = st.session_state.chat_input

    if 'OPENAI_API_KEY' not in os.environ:
        st.write("You need an OpenAI API Key to chat.")
        return

    if not chat_input:
        st.write("Please enter a message to send.")
        return

    if chat_input: # if the user entered a question and hit enter
        vector_store = st.session_state.vs if 'vs' in st.session_state else None

        answer = ask_and_get_answer(MODEL_NAME, vector_store, chat_input, st.session_state.k_value)

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = INITIAL_CHAT_HISTORY

        st.session_state.chat_history.append({"role": "user", "content": st.session_state.chat_input})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        for message in reversed(st.session_state.chat_history[1:]):
            with st.chat_message(message["role"]):
                st.write(message["content"])


# clear the chat history from streamlit session state
def clear_history():
    st.session_state.chat_history = INITIAL_CHAT_HISTORY


if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.set_page_config(layout="wide")
    with st.sidebar:
        st.subheader('Chat With Your AI')

        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        api_key = st.text_input("OpenAI API Key:", type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            
        # file uploader widget
        uploaded_files = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'], accept_multiple_files=True)

        # chunk size number widget
        chunk_size = st.slider('Chunk size:', min_value=100, max_value=2048, value=512, step=10, on_change=clear_history)

        # chunk overlap number widget
        chunk_overlap = st.slider('Chunk overlap:', min_value=0, max_value=2048, value=20, step=10, on_change=clear_history)

        # k number input widget
        k = st.slider('k', min_value=1, max_value=20, value=3, step=10, on_change=clear_history, key="k_value")
        
        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_files and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):
                all_chunks = []

                for file in uploaded_files:
                    # writing the file from RAM to the current directory on disk
                    bytes_data = file.read()
                    file_name = os.path.join('./', file.name)
                    with open(file_name, 'wb') as f:
                        f.write(bytes_data)

                    data = load_document(file_name)
                    all_chunks += chunk_data(data, chunk_size, chunk_overlap)
                
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(all_chunks)}')
                tokens, embedding_cost = calculate_embedding_cost(MODEL_NAME, all_chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(all_chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')

    msg = st.chat_message("assistant")
    msg.write(INITIAL_CHAT_HISTORY[0]["content"])
    
    st.chat_input('Say something', key="chat_input", on_submit=send_chat_msg)
