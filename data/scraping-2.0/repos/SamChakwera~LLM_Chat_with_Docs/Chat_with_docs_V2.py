
import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from PIL import Image

def load_document(file):
    """Load document from file based on its extension."""
    loader_map = {'.pdf': PyPDFLoader, '.docx': Docx2txtLoader, '.txt': TextLoader}
    name, extension = os.path.splitext(file)
    try:
        print(f'Loading {file}')
        loader = loader_map[extension](file)
    except KeyError:
        print('Document format is not supported!')
        return None
    return loader.load()

def chunk_data(data, chunk_size=256, chunk_overlap=20):
    """Chunk the data into smaller pieces."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(data)

def create_embeddings(chunks):
    """Create embeddings for the chunks."""
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(chunks, embeddings)

def ask_and_get_answer(vector_store, q, k=3):
    """Ask a question and get answer."""
    llm = ChatOpenAI(model='gpt-4', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return chain.run(q)

def calculate_embedding_cost(texts):
    """Calculate the embedding cost."""
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.0004

def clear_history():
    """Clear the history in the session state."""
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=True)

    hide_default_format = '''
           <style>
           #MainMenu {visibility: hidden; }
           footer {visibility: hidden;}
           </style>
           '''
    st.set_page_config(page_title="LLM QA Chatbot")
    st.markdown(hide_default_format, unsafe_allow_html=True)

    st.image('img.png', width=300)
    st.subheader('LLM QA Chatbot by Samuel Chakwera')
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        upload_file = st.file_uploader('Upload a document:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk Size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        add_data = st.button('Add Data', on_click=clear_history)

        if upload_file and add_data:
            with st.spinner('Reading, chunking and embedding file ...'):
                bytes_data = upload_file.read()
                file_name = os.path.join('./', upload_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                if data is not None:
                    chunks = chunk_data(data, chunk_size=chunk_size)
                    st.write(f'Chunk size: {chunk_size}, Number of chunks: {len(chunks)}')
                    tokens, embbeding_cost = calculate_embedding_cost(chunks)
                    st.write(f'Embedding Cost in USD: {embbeding_cost:.4f}')

                    vector_store = create_embeddings(chunks)

                    st.session_state.vs = vector_store
                    st.success('File uploaded, chunked and embedded successfully!')

    q = st.text_input('Ask a question about the content of your file:')
    if q and 'vs' in st.session_state:
        vector_store = st.session_state.vs
        st.write('Searching for answer ...')
        answer = ask_and_get_answer(vector_store, q, k)
        if answer is not None:
            st.text_area('LLM Answer:', value=answer, height=200)

    st.divider()
    if 'history' not in st.session_state:
        st.session_state.history = ''
    if 'answer' in locals() and answer is not None:
        value = f'Your Question:  {q} \nAnswer:  {answer}'
        st.session_state.history = f'{value} {"-"*100} \n {st.session_state.history}'
    st.text_area('History:', st.session_state.history, height=400)

st.markdown(
    '''<footer>
    Created by <a href="https://stchakwera.com/">Samuel Chakwera</a>
    </footer>''',
    unsafe_allow_html=True
)
