from dotenv import load_dotenv
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import  ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import os
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.llms import OpenAI
import tempfile
import json

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            fp.write(uploaded_file.getvalue())
            return fp.name
    except Exception as e:
        print(f"Error: {e}")
        return None

def save_chat_history(chat_history, filename):
    with open(filename, 'w') as f:
        json.dump(chat_history, f)

def load_chat_history(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
        # Getting the file_path
        file_path = save_uploaded_file(pdf)

        # Loading the pdf documents from the path
        loader = PyPDFLoader(file_path)

        # Getting the content of the document
        documents = loader.load()
        
        # Spiltting the document into Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Getting the splits
        docs = text_splitter.split_documents(documents)
        
        # Removing the existing files
        os.system("rm -rf docs/chroma/*")

        # Directory to store the text embeddings
        persist_directory = 'docs/chroma/'

        # Creating the embedding object
        embedding = OpenAIEmbeddings()

        # Creating the vector database
        vectordb = Chroma.from_documents(documents=docs, persist_directory=persist_directory, embedding=embedding)

        # Creating the LLM Chain
        llm = OpenAI(temperature=0)

        # Creating a compressor, this is going to help us compress the document retrieved for answering a given query
        # This is particularly helpful because we only pass the relevant information to the LLM to answer the question
        compressor = LLMChainExtractor.from_llm(llm)

        # Creating a retriever from the base compressor and retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectordb.as_retriever(search_type = "mmr", search_kwargs={"k": 3})
        )
    
        # Selecting model type
        model_type = st.selectbox('Select model type', ['gpt-4', 'gpt-3.5-turbo-0301'], index=1)

        # Chain type
        chain_type = st.selectbox('Select chain type', ['stuff', 'map_reduce', 'refine', 'map_rerank'], index=0)

        # Compression
        use_compression = st.checkbox('Use Compression in retrieval', value=False)

        # Checkbox to decide whether to return source documents.
        return_source = st.checkbox('Return source documents', value=True) 

        # Checkbox to decide whether to return generated question.
        return_generated_question = st.checkbox('Return generated question', value=True) 

        # show user input
        user_question = st.text_input("Ask a question about your PDF:", value="")

        # Loading chat history 
        chat_history = load_chat_history('chat_history.json')
        chat_history = [tuple(x) for x in chat_history]

        if user_question:
            # Memory is managed externally.
            # If we're passing a memory then we can't look at the source documents
            # In that case, we'll have to pass the chat history separately
            retriever = compression_retriever if use_compression else vectordb.as_retriever(search_type = "mmr", search_kwargs={"k": 3})

            qa = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(model_name=model_type, temperature=0), 
                condense_question_llm = ChatOpenAI(temperature=0, model=model_type),
                return_source_documents=return_source,
                return_generated_question=return_generated_question,
                chain_type=chain_type,
                retriever=retriever,
                verbose=True
            )

            api_call_metadata = None

            with get_openai_callback() as cb:
                response = qa({"question": user_question, "chat_history": chat_history})
                api_call_metadata = cb
            
            # Add a button
            if st.button('Clear Chat History'):
                # This block runs when the button is clicked.
                # So you just clear your chat history here.
                chat_history = []
                save_chat_history(chat_history, 'chat_history.json')

            # Write chat history to the page
            with st.expander("See Chat History"):
                for i, chat in enumerate(chat_history):
                    st.write(f"Q{i+1}: {chat[0]}")
                    st.write(f"A{i+1}: {chat[1]}")

            # Showing the generated question
            if return_generated_question:
                with st.expander("See Generated Question"):
                    st.write(response['generated_question'])
            
            # Writing the output
            st.write(response['answer'])

            # Showing the source documents
            if return_source:
                with st.expander("See Source Documents"):
                    st.write("\n\n".join([x.page_content for x in response['source_documents']]))

            # Showing the API call metadata
            with st.expander("See API Call Metadata"):
                    st.write(api_call_metadata)

            # Add new question to chat history
            chat_history.append((user_question, response['answer']))
            
            # Save chat history to file
            save_chat_history(chat_history, 'chat_history.json')

if __name__ == '__main__':
    main()