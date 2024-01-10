from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import streamlit as st
import os
import pickle
from PyPDF2 import PdfReader

def st_siderbar():
    with st.sidebar:
        st.title('AskPDF App')
        st.write('')
        option_text_splitter = st.radio(
            "Choose a Text splitter",
            key="optTextSplitter",
            options=["CharacterTextSplitter", "RecursiveCharacterTextSplitter"],
            index=0,
        )
        option_vector_store = st.radio(
            "Choose a Vectore store",
            key="optVectorStore",
            options=["FAISS", "FAISS(disk)", "Chroma", "Chroma(disk)"],
            index=0,
        )
        st.write('')
        st.markdown('''
        ## About
        This app is an LLM-powered askPDF built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) LLM model    
        ''')
    return option_text_splitter, option_vector_store

# Reference : https://docs.langflow.org/components/text-splitters
def text_split_by_CharacterTextSplitter():
    return CharacterTextSplitter(
        chunk_size=1500,            # Define the maximum size of each text chunk [default : 1000]
        chunk_overlap=100,          # Define the number of characters for overlapping between chunks [default : 200]
        length_function=len,        # Function used to calculate the length of the text
        separator="\n"              # Specify the character to use for splitting the text (newline in this case) [default : "."]
    )

def text_split_by_RecursiveCharacterTextSplitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
        separators=["\n", ".", " "] # Defaults to ["\n\n", "\n", " ", ""]
    )

def process_split_embedding_vectorize(pdf_text, vectorStore_space, option_text_splitter, option_vector_store):
    # Split text into manageable chunks
    if option_text_splitter == "CharacterTextSplitter":
        text_splitter = text_split_by_CharacterTextSplitter()
    else:
        text_splitter = text_split_by_RecursiveCharacterTextSplitter()

    chunks = text_splitter.split_text(text=pdf_text)                    # Split the text into chunks using the specified parameters
    # print(">> chunks count : ", len(chunks))
    
    # Create embeddings for the text chunks
    embeddings = OpenAIEmbeddings()                                     # Initialize an embeddings object using OpenAI's API

    # Vector Store
    if option_vector_store == "FAISS":                                  # https://python.langchain.com/docs/integrations/vectorstores/faiss
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

    elif option_vector_store == "FAISS(disk)":
        # Create a FAISS index from the text chunks and their embeddings
        # https://stackoverflow.com/questions/77605224/cannot-pickle-thread-rlock-object-while-serializing-faiss-object
        # vectorStore_space = pdf.name[:-4]
        
        if os.path.exists(vectorStore_space):
            VectorStore = FAISS.load_local(vectorStore_space, embeddings=embeddings)
            st.write(":blue[Embeddings loaded from your file already (disks)]")
        else:
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            VectorStore.save_local(vectorStore_space)
            st.write(":orange[Embedding computation completed]")
                
    elif option_vector_store == "Chroma":
        # load it into Chroma
        VectorStore = Chroma.from_texts(chunks, embedding=embeddings)

    elif option_vector_store == "Chroma(disk)":
        vectorStore_space = f"{vectorStore_space}.chroma"
        
        if os.path.exists(vectorStore_space):
            VectorStore = Chroma(persist_directory=vectorStore_space, embedding_function=embeddings)
            st.write(":blue[Embeddings loaded from your file already (disks)]")
        else:
            VectorStore = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=vectorStore_space)
            VectorStore.persist()
            st.write(":orange[Embedding computation completed]")
                
    else:
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

    return VectorStore

def main():    
    st.set_page_config(                 # Define Streamlit page configuration.
        page_title="Ask your PDF",      # Title of the page.
        page_icon="⌨️"                   # Icon of the page.
    )
    st.header("Ask your PDF")           # Set the header text of the Streamlit web page
    
    # Sidebar contents
    (option_text_splitter, option_vector_store) = st_siderbar()
        
    # Upload file section
    pdf = st.file_uploader("Upload your PDF", type="pdf")  # Create a file uploader widget specifically for PDF files
    cancel_button = st.button('Cancel')
    if cancel_button:
        st.stop()
            
    # Extract text from the PDF
    if pdf is not None:                     # Check if a PDF file has been successfully uploaded
        pdf_reader = PdfReader(pdf)         # Initialize a PDF reader to read the uploaded file
        pdf_text = ""                           # Initialize a string to accumulate extracted text
        for page in pdf_reader.pages:       # Loop through each page in the PDF
            pdf_text += page.extract_text()     # Append the extracted text from each page to the 'text' variable

        # Load environment variables from a .env file, useful for hiding sensitive information
        load_dotenv()

        # Create the knowledge base(VectorStore) object
        vectorStore_space = pdf.name[:-4]
        VectorStore = process_split_embedding_vectorize(pdf_text, vectorStore_space, option_text_splitter, option_vector_store)
        
        # User input for querying the PDF
        user_question = st.text_input("Ask a question about your PDF: ")    # Create an input field for the user to ask a question
        if user_question:                                                   # Check if the user has entered a question
            docs = VectorStore.similarity_search(query=user_question, k=1)  # Perform a similarity search in the knowledge base for the user's question
            # print(">> docs : ", docs)
            
            llm = OpenAI()                                                  # Initialize the OpenAI language model
            chain = load_qa_chain(llm=llm, chain_type="stuff")              # Load a QA chain for processing the query and documents
                                                                            # Documents Docs : https://python.langchain.com/docs/modules/chains/document/
                                                                            # . stuff : The stuff documents chain is the most straightforward of the document chains. It takes a list of documents, inserts them all into a prompt and passes that prompt to an LLM.
                                                                            # . refine : The Refine documents chain constructs a response by looping over the input documents and iteratively updating its answer
                                                                            # . Map Reduce : The map reduce documents chain first applies an LLM chain to each document individually (the Map step), treating the chain output as a new document.
                                                                            # . Map Re-rank : The map re-rank documents chain runs an initial prompt on each document, that not only tries to complete a task but also gives a score for how certain it is in its answer
            with get_openai_callback() as cb:                               # Use a context manager for handling callbacks
                response = chain.run(input_documents=docs, question=user_question)  # Run the chain to get a response for the user's question
                # print(cb)

            st.write(response)

if __name__ == '__main__':
    main()
