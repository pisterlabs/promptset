import os
import asyncio
import textwrap
import time
import streamlit as st
import langchain
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.document_loaders import PyPDFLoader

def main():
    os.environ['OPENAI_API_KEY'] = 'aade9d18f8a241cca31b5c44d60cc7fa'
    OPENAI_API_KEY = 'aade9d18f8a241cca31b5c44d60cc7fa'
    os.environ['OPENAI_API_TYPE'] = 'azure'
    os.environ['OPENAI_API_VERSION'] = '2023-09-15-preview' # api version
    os.environ['AZURE_OPENAI_ENDPOINT'] = 'https://gpt4-for-bot.openai.azure.com/' # open api base
    os.environ['OPENAI_DEPLOYMENT_NAME'] = 'gpt4-for-bot'
    print("Program Started")
    
    # Setting up app title
    st.title('My Lang Chatbot')
    
    # loader = UnstructuredFileLoader('Avalanche622.pdf')
    from langchain.document_loaders import DirectoryLoader
    loader = DirectoryLoader(path='F:\AzureModels\MyDocs',glob='./*.pdf',loader_cls=PyPDFLoader)
    documents = loader.load()
    print('Document loaded')
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print('Text Loaded from Text Splitter')
    
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # load it into Chroma
    db = Chroma.from_documents(texts, embedding_function)
    print('Texts loaded in DB')
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    print("Retriver Set")
    lllm =  AzureOpenAI(
        model_name="gpt-35-turbo",
        deployment_name='gpt4-for-bot')
    chain = RetrievalQA.from_chain_type(llm=lllm,
        chain_type= 'stuff',
        retriever = retriever,
        return_source_documents=True)
    print("LLM and Chain set!")
    print("Program Ready to Chat")
    
    def process_llm_response(response):
        lines = response['result'].split('\n')
        wrapped_lines = [textwrap.fill(line, width=110) for line in lines]
        wrapped_text = '\n'.join(wrapped_lines)
        source = response['source_documents'][0].metadata['source'].split('/')[-1]
        return wrapped_text,source
    
    #####################################################
    # Setting Up Streamlit
    #####################################################
        
    ### Adding a Funcationality to Select a Model
    # Models_list= st.radio(
    #     "Select a Model you want to test",
    #     ["GPT-3.5-Turbo", "***GPT4***", "FastChat","falcon-40b"],index=None)
    # st.write("You selected:", Models_list)
    
    # Setup a session state message variable to hold all the old messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display all the historical messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    
    # Build a prompt input template to display the prompts
    prompt = st.chat_input('Pass Your Prompt here')
    
    async def call_chain(query):
        result = await chain.acall(query)
        return result
    
    
    # If the user hits enter then
    if prompt:
        # Display the prompt
        st.chat_message('user').markdown(prompt)
        # Store the user prompt in state
        #st.session_state.messages.append({'role':'user','content':prompt})
        # Send the prompt to the llm
        response = asyncio.run(call_chain(prompt))
        response,source = process_llm_response(response)
    # Show the llm response
    st. chat_message('assistant').markdown(response)
    st. chat_message('assistant').markdown(source)
    # Store the LLM Response in state
    st.session_state.messages.append({'role':'user','content':prompt})

if __name__ == "__main__":
    main()
