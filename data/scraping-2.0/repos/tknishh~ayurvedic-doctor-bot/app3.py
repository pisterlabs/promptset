import streamlit as st
import glob
import os
import pdfplumber  # Import pdfplumber
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

# 1. Vectorize the responses from PDF files
pdf_directory = "path_to_pdf_directory"  # Update with your directory path
pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))

# Create a function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Load text from all PDF files
documents = [extract_text_from_pdf(pdf_file) for pdf_file in pdf_files]

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)


# 2. Function for similarity search


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    return page_contents_array


# 3. Setup LLMChain & prompts

repo_id = "google/flan-t5-xxl"

llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}
)

template = """
You will help me provide Ayurvedic advice for various health concerns.
I will share a user's message with you, and your task is to provide the most suitable Ayurvedic guidance based on past responses.
Please adhere to the following guidelines:

1/ Your response should closely resemble or even match previous responses.

2/ If previous responses are not directly applicable, try to maintain a style consistent with past interactions.

Here is a message I received from the user:
{message}

Below, you'll find a list of past responses that we've used in similar situations:
{past_responses}

Kindly compose the most appropriate Ayurvedic advice to send to this user:
"""

prompt = PromptTemplate(
    input_variables=["message", "past_responses"], template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation

def generate_response(message):
    past_responses = retrieve_info(message)
    response = chain.run(message=message, past_responses=past_responses)
    return response


# 5. Build an app with streamlit
def main():
    st.set_page_config(page_title="Medical Chatbot", page_icon=":books:")

    st.header("Medical Chatbot :book:")
    message = st.text_area("user query")

    if message:
        st.write("Generating best advice...")

        result = generate_response(message)

        st.info(result)

if __name__ == "__main__":
    main()
