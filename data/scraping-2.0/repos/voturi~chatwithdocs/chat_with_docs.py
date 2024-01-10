import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os


def load_document(file):
    """Loading documents of various formats"""
    if file.endswith('.txt'):
        from langchain.document_loaders import TextLoader
        print(f'Loading File....{file}')
        loader = TextLoader(file)
        data = loader.load()
        return data
    elif file.endswith('.pdf'):
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading File....{file}')
        loader = PyPDFLoader(file)
        data = loader.load()
        return data
    elif file.endswith('.docx'):
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
        data = loader.load()
        return data
    elif file.endswith('.doc'):
        from langchain.document_loaders import DocLoader
        print(f'Loading File....{file}')
        loader = DocLoader(file)
        data = loader.load()
        return data

    return data


def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask_and_get_answers(vector_store, query, k=5):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo')
    retriever = vector_store.as_retriever(
        search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(
        retriever=retriever, chain_type='stuff', llm=llm)
    answer = chain.run(query)
    return answer


def calculate_embeddings_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('gpt-3.5-turbo')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Cost : {total_tokens*0.00000006} USD')
    return total_tokens, total_tokens*0.00000006


def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


def clear_input():
    if 'query' in st.session_state:
        del st.session_state['query']


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    # st.image('img.png')
    st.title('Chat with Documents')
    # st.subheader('upload and ask questions about your documents')
    with st.sidebar:
        api_key = st.text_input('Enter API Key', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        uploaded_file = st.file_uploader(
            "Choose a file", type=['txt', 'pdf', 'docx', 'doc'])

        chunk_size = st.number_input(
            'Enter Chunk Size', min_value=100, max_value=2048, value=512, on_change=clear_history)

        k = st.number_input('k', min_value=1, max_value=20,
                            value=3, on_change=clear_history)

        add_data = st.button('Add Data')

        if uploaded_file and add_data:
            with st.spinner('Loading Data...chunking and embedding...'):
                byte_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(byte_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(
                    f'Chunk size: {chunk_size} , Number of Chunks: {len(chunks)}')

                total_tokens, cost = calculate_embeddings_cost(chunks)
                st.write(
                    f'Total Tokens: {total_tokens} , Cost: {cost:.4f} USD')

                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store

                st.success('File uploaded, chunked and embedded Successfully')

    query = st.text_input(
        'Enter your question about the content of your document:', key='query')
    if query:
        if 'vs' in st.session_state:
            with st.spinner('Searching for answers...'):
                answer = ask_and_get_answers(st.session_state.vs, query, k=k)
                st.success(f'Answer: {answer}')
        else:
            st.error('Please upload a document first')

        st.divider()

        if 'history' not in st.session_state:
            st.session_state.history = ''

        value = f'Q: {query}\nA: {answer}\n'
        st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
        st.text_area('Chat History',
                     value=st.session_state.history, key='history', height=400)

    st.write('voturi@gmail.com')
