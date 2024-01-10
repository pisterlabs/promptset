import streamlit as st
import config
import os
import PyPDF2
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import tempfile


# TO DO (REMAINING)
# -> ADD CHECKBOX FOR FILES
# -> SOLVE MULTIPLE RUNNING ISSUE

uploaded_files = st.sidebar.file_uploader("Upload a document", type=["txt", "pdf", "docx", "csv"], accept_multiple_files=True)

# Create a list to store the filenames and their corresponding checkboxes
file_list = []

st.sidebar.write("Select the checkbox for every file you want to use.")
# Run loop to read every file
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        # Get the filename
        filename = uploaded_file.name

        # Create a checkbox for each file and add it to the file_list
        checkbox = st.sidebar.checkbox(filename, key=filename)
        file_list.append((filename, checkbox))

# Filter the selected files from the file_list
selected_files = [filename for filename, checkbox in file_list if checkbox]

# Run loop to process the selected files
if selected_files:
    text = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name in selected_files:
            if uploaded_file.type == "application/pdf":
                # Create a temporary file to store the uploaded file
                temp_file = tempfile.NamedTemporaryFile(delete=False)

                # Extract text from the PDF file using PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                pdf_text = ""
                for page in range(len(pdf_reader.pages)):
                    pdf_text += pdf_reader.pages[page].extract_text()

                temp_file.write(pdf_text.encode('utf-8'))
                temp_file.close()

                # Create an instance of the TextLoader
                loader = TextLoader(file_path=temp_file.name)

                # Load the document using the TextLoader
                document = loader.load()

                # Splitting text with size 1000 tokens
                textsplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
                chunks = textsplitter.split_documents(documents=document)  # list of chunks

                text.extend(chunks)  # appending chunks to main text list

                os.remove(temp_file.name)  # removing temporary file

            else:
                # Create a temporary file to store the uploaded file
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(uploaded_file.read())
                temp_file.close()

                # Create an instance of the TextLoader
                loader = TextLoader(file_path=temp_file.name)

                # Load the document using the TextLoader
                document = loader.load()

                # Splitting text with size 1000 tokens
                textsplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
                chunks = textsplitter.split_documents(documents=document)  # list of chunks

                text.extend(chunks)  # appending chunks to main text list

                os.remove(temp_file.name)  # removing temporary file

    if text:
        print("Text List:", text)  # debug

        # Splitting text with size 1000 tokens
        textsplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
        chunks = textsplitter.split_documents(documents=text)  # list of chunks

        # Embedding
        embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)

        # Storing in database
        db = Chroma.from_documents(text, embeddings)  # Creating vector store

        retriever = db.as_retriever()  # Retrieve relevant data from vector space
        # make a small chunk of the dataset based on similarity and then pass to retriver.
        
        question_answer = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=config.OPENAI_API_KEY), chain_type="stuff", retriever=retriever)

        question = str(st.text_input("Enter your question (q to quit)", key="question1"))

        keyno = 2
        if not question.strip():
            print("You have not entered a question.")

        while question.lower() != 'q' and question != "":
            if not question.strip():
                print("You have not entered a question.")
            answer = question_answer(question)
            st.write("Answer:", answer['result'])
            keyvar = "question" + str(keyno)
            question = st.text_input("Enter your question (q to quit)", key=keyvar)
            keyno += 1
        if question.lower() == 'q':
            st.write("Thank you for using this application! Have a nice day.")
            SystemExit(0)
    else:
        st.write("No files uploaded.")
else:
    st.write("No files uploaded.")

"""
1. use a local database to store the embeddings
2. check line 105
3. PDFMiner

"""