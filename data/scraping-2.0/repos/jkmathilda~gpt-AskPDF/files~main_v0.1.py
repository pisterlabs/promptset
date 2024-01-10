from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import streamlit as st
from PyPDF2 import PdfReader

def main():
    load_dotenv()                                   # Load environment variables from a .env file, useful for hiding sensitive information
    st.set_page_config(page_title="Ask your PDF")   # Configure the Streamlit page with a specific title
    st.header("Ask your PDF")                       # Set the header text of the Streamlit web page
    
    # Upload file section
    pdf = st.file_uploader("Upload your PDF", type="pdf")  # Create a file uploader widget specifically for PDF files
            
    # Extract text from the PDF
    if pdf is not None:                     # Check if a PDF file has been successfully uploaded
        pdf_reader = PdfReader(pdf)         # Initialize a PDF reader to read the uploaded file
        text = ""                           # Initialize a string to accumulate extracted text
        for page in pdf_reader.pages:       # Loop through each page in the PDF
            text += page.extract_text()     # Append the extracted text from each page to the 'text' variable

        # Split text into manageable chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",         # Specify the character to use for splitting the text (newline in this case)
            chunk_size=1000,        # Define the maximum size of each text chunk
            chunk_overlap=200,      # Define the number of characters for overlapping between chunks
            length_function=len     # Function used to calculate the length of the text
        )
        chunks = text_splitter.split_text(text)  # Split the text into chunks using the specified parameters

        # Create embeddings for the text chunks
        embeddings = OpenAIEmbeddings()                         # Initialize an embeddings object using OpenAI's API
        knowledge_base = FAISS.from_texts(chunks, embeddings)   # Create a FAISS index from the text chunks and their embeddings

        # User input for querying the PDF
        user_question = st.text_input("Ask a question about your PDF: ")    # Create an input field for the user to ask a question
        if user_question:                                                   # Check if the user has entered a question
            docs = knowledge_base.similarity_search(user_question)          # Perform a similarity search in the knowledge base for the user's question
            
            llm = OpenAI()                                                  # Initialize the OpenAI language model
            chain = load_qa_chain(llm, chain_type="stuff")                  # Load a QA chain for processing the query and documents
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
