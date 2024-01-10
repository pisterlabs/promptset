import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import openai
from openai import ChatCompletion

openai.api_type = "azure"
openai.api_base = "https://ausopenai.azure-api.net"
openai.api_version = "2023-07-01-preview"
openai.api_key = "6693a6eec2eb4b9b9f4ff83d5809fb36"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader("investmentbanking.pdf")
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def main():

    st.set_page_config(page_title="Chat with Investment Banking",
                       page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("GPT Investment Banking")
    prompt = st.text_input("Ask a question about Investment Banking:")

    pdf_docs = "investmentbanking.pdf"
    if st.button("Submit Query"):
        with st.spinner("Processing"):
            # get pdf text
            raw_text = get_pdf_text(pdf_docs)
            response = openai.ChatCompletion.create(
            engine="gpt-35-turbo-16k",
            messages=[{ "role": "system", "content": raw_text + "\n" + prompt }, { "role": "user", "content": "" }
                        ]
            )
            st.write(response['choices'][0]['message']['content'])

if __name__ == '__main__':
    main()