import os
import nltk
import faiss
import numpy as np
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain import PromptTemplate
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

st.title("MY PDF ANSWERING MACHINE")
st.text("(Upload your PDF -- Ask a question)")

load_dotenv()
nltk.download('punkt')

@st.cache_resource
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

target_dir = "pdfs"

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

def generate_embeddings(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    model = load_model()
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return embeddings, sentences

def create_faiss_index(embeddings):
    d = embeddings.shape[1]  
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.cpu().detach().numpy())  
    return index

def search_in_index(index, query, sentences, top_k=5):
    model = load_model()
    query_embedding = model.encode([query], convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().detach().numpy()
    distances, indices = index.search(query_embedding_np, top_k)

    results = [(sentences[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results

if uploaded_file:
    try:
        pdf_reader = PdfReader(uploaded_file)
        extracted_text = ""
        for page in pdf_reader.pages:
            extracted_text += page.extract_text()

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.1)

        template = """
        As an expert in extracting relevant and meaningful information, your task is to process the input provided. The input contains a user's query followed by a text. They are separated by the phrase 'End of Query'. Carefully understand the query to extract the relevant information from the text.
        "Note that the text that you may have to read to extract query may be disorganized. You need to use your intelligence to understand the same and properly extract information".

        Input Provided:
        {input_str}

        Based on the user's query, provide precise information related to it from the text. Present the information in bullet points, if possible.

        \n\n"""

        prompt_template = PromptTemplate(input_variables=["input_str"], template=template)
        question_chain = LLMChain(llm=llm, prompt=prompt_template)

        overall_chain = SimpleSequentialChain(
            chains=[question_chain]
        )

        text = st.text_input('Enter your question here ')
        if text:
            st.write("Response : ")
            with st.spinner("Searching for answers ..... "):
                embeddings, sentences = generate_embeddings(extracted_text)
                index = create_faiss_index(embeddings)
                results = search_in_index(index, text, sentences)
                txt = ''
                for sentence, score in results:
                    text+=sentence
                input_str = f"query: {text}\nEnd of Query\ntext: {txt}"
                result = overall_chain.run(input_str)
                st.write(result)
            st.write("")
        

    except Exception as e:
        st.write(e)