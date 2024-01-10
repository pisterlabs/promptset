import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os

def load_document(file):
    import os
    name, extension = os.path.splitext(file)
    
    print(f'Loading {file}...')
    if extension == ".pdf":
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
    elif extension == ".docx":
        from langchain.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    elif extension == ".txt":
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print(f'Document extension {extension} is not supported.')
        
    data = loader.load()
    return data

def chunk_data(data,chunk_size=256,chunk_overlap = 20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks
    
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks,embeddings)
    return vector_store

# higher the k value the pricier it will be as you'll 
# use more tokens to get more accurate answer.
def ask_and_get_answer(vector_store,q,k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity',search_kwargs={'k':k})

    chain = RetrievalQA.from_chain_type(llm=llm,chain_type='stuff',retriever=retriever)
    
    answer = chain.run(q)
    
    return answer

def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    price = total_tokens/1000 * 0.0004
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens/1000 * 0.0004:.6f}')
    return total_tokens,price

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    
    st.image('image.png')
    st.divider()
    st.subheader('LLM Question-Answering Application 🤖')
    os_api_key = os.environ.get('OPENAI_API_KEY') or ""
    
    
    with st.sidebar:
        api_key = st.text_input('Open API key:', type='password',value=os_api_key)
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        uploaded_file = st.file_uploader('Upload a file:',type=['pdf','docx','txt'])
        chunk_size = st.number_input('Chunk size:',min_value=100,max_value=2048,value=512,on_change=clear_history)
        k = st.number_input('k', min_value=1,max_value=20,value=3,on_change=clear_history)
        add_data = st.button('Add Data',on_click=clear_history)

        if uploaded_file and add_data:
            if  api_key =="":
                st.error("Please enter a valid Open Api Key.",icon="🚫")
            else:
                with st.spinner('Reading, chunking and embedding file ...'):
                    bytes_data = uploaded_file.read()
                    file_name = os.path.join('./',uploaded_file.name)

                    with open(file_name, 'wb') as f:
                        f.write(bytes_data)
                    
                    data = load_document(file_name)
                    chunks = chunk_data(data,chunk_size=chunk_size)
                    st.write(f'Chunk Size:{chunk_size}, Chunks:{len(chunks)}')

                    tokens, embedding_cost = calculate_embedding_cost(chunks)
                    st.write(f'Embedding Cost:${embedding_cost:.4f}')

                    vector_store = create_embeddings(chunks)
                    st.session_state.vs = vector_store

                    st.success('File Uploaded, chunked and embedded successfully.')

    q = st.text_input('Ask a question about the content of your file:',disabled= api_key == "")
    if q:
        if api_key == "":
            st.error("Please enter a valid Open Api Key.",icon="🚫")
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            answer = ask_and_get_answer(vector_store,q,k)
            st.text_area('LLM Answer: ',value=answer);
    
            st.divider()

            if 'history' not in st.session_state:
                st.session_state.history=''
            
            value = f'Q: {q} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n { st.session_state.history}'
            h = st.session_state.history
            st.text_area(label="Chat History", value =h,key="history", 
                         height=400,placeholder="Search History is empty...")