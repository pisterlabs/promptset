import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Set API keys
os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "28c2445d1bfe7530595be6fbc858b1968d776af69c6034aa5feda50deab4b990"

# PDF processing
pdfreader = PdfReader('XYZ_contract_pdf_Sumit Yenugwar 4.pdf')
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# Text splitting
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

# Create document search
document_search = FAISS.from_texts(texts, embeddings)

# Load QA chain
chain = load_qa_chain(OpenAI(), chain_type="stuff")

with st.sidebar:
    st.title('🤗💬 LLM Chat APP')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    st.markdown("<br>", unsafe_allow_html=True)  # Add vertical space
    st.write('Made with ❤️ by [Prompt Engineer](https://www.youtube.com/watch?v=M4mc-z_K1NU&list=PLUTApKyNO6MwrOioHGaFCeXtZpchoGv6W)')

# Streamlit app
def main():
    st.title("DAMA-Data Management body of knowledge")

    # Text input area
    user_input = st.text_area("Enter your MCQ question ",height=150)

    # Button to trigger model inference
    if st.button("Get Answer"):
        # Combine user input with the prompt and query
        prompt_query = f"you have provided with MCQ question and its option as a chatbot model: {user_input}"
        text_query = prompt_query + user_input
        # Perform similarity search
        docs = document_search.similarity_search(text_query)

        # Run the model with the combined text and query
        model_answer = chain.run(input_documents=docs, question=user_input)

        # Display the model's answer
        st.text_area("Model Answer:", value=model_answer)

# Run the Streamlit app
if __name__ == "__main__":
    main()
