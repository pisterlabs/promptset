# za funkcije u requirements.txt staviti https://github.com/yourusername/public-repo

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredPDFLoader
import streamlit as st

dokum = st.file_uploader(
    "Izaberite dokument/e", key="upload_file", type=['txt', 'pdf', 'docx'])

namespace = st.text_input('Unesi naziv namespace-a: ')
submit_button = st.form_submit_button(label='Submit')
if submit_button and not dokum == "" and not namespace == "":
    # Load the file
    if ".pdf" in dokum.name:
        loader = UnstructuredPDFLoader(
            dokum.name, encoding="utf-8")
    else:
        # Creating a file loader object
        loader = UnstructuredFileLoader(
            dokum.name, encoding="utf-8")

data = loader.load()  # Loading text from the file

# Split the document into smaller parts
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(data)

# Store the texts in a .txt file
with open('output.txt', 'w', encoding='utf-8') as file:
    for text in texts:
        # Write each text segment to the file on a new line
        file.write(text + '\n')


# Read the texts from the .txt file
stored_texts = []
with open('output.txt', 'r', encoding='utf-8') as file:
    for line in file:
        # Remove leading/trailing whitespace and add to the list
        stored_texts.append(line.strip())

# Now, you can use stored_texts as your texts
