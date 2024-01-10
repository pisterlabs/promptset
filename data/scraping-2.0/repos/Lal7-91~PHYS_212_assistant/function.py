import os
import openai
import streamlit as st

import json


from PyPDF2 import PdfReader
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import faiss
FAISS = faiss.FAISS

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
#os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']




def load_and_extract_one_pdf(pdf_file_name):
    

    pdf_file_obj = open(os.path.join(os.getcwd(), "data", pdf_file_name), "rb")
    pdf_reader = PdfReader(pdf_file_obj)

    extracted_text = ""
    for page in pdf_reader.pages:
        extracted_text += page.extract_text()

    pdf_file_obj.close()
    return extracted_text



def Summary(text):
    
    promopt = f"""

        You are an expert Physics summrizer. You will be given a topic delimited by four backquotes, 
        Make sure to capture the main points, key arguments, and any supporting evidence presented in the topic.
        Your summary should be informative and well-structured, 3-6 sentences for each main point and use point method if neded. 
        Also use bold words if need and spaces. make sure that is Complete and simple to understand and in best form. Also add the Formulas. use points and short pargraph. 

        text = ''''{text}''''
        
        """


    response = openai.ChatCompletion.create(
        
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": promopt
            }
        ]
    )
    return response['choices'][0]['message']['content']
    


class make_quiz:
    

    TEMPLATE = """questions": [
        {
            "id": 1,
            "question": "what is the purpos of assembler directives?",
            "options": [
                "A. To define segments and allocate space for varibles",
                "B. To represent specific machine instructions",
                "C. To simplify the programmer`s task",
                "D. To prove information to the assembler"

            ],
            "correct_answer": "D. To prove information to the assembler"
        }]"""


    def __init__(self, model="gpt-3.5-turbo-16k"):
        self.model = model

    def display_questions(self, questions):
        """Displays the given questions in a streamlit-friendly format."""

        for question in questions:

            question_id = str(question["id"])

            st.write(
                f"## Q{question_id} \ {question['question']}"
            )

            options_text = ""
            options = question["options"]

            for option in options:
                options_text += f"- {option}\n"

            st.write(options_text)

            with st.expander("show answer", expanded=False):
                st.write(question["correct_answer"])

            st.divider()

        st.subheader("End of questions")

    def get_questions(self, text):
        
        prompt = f"""
            You are a expert in physics. create a 10 multiple-choice questions (MCQS) based on the text delimted by four backquotes, 
            4 of them definition and the other 6 math qustions from the topic formulas and Numerical equations, 
            try to make them in Multiple difficulties 
            the response must be formatted in JSON. Each question contains id, question, options as list, correct_answer.
            this is an example of the response: {self.TEMPLATE}
            the text is : '''' {text}''''

            """

        response = openai.ChatCompletion.create(
            model=self.model
            , messages=[
                {"role": "system",
                  "content": prompt
                  }
                ]
            )

        return json.loads(response["choices"][0]["message"]["content"])



def creat_docs(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 2000, chunk_overlap= 200)
    docs = text_splitter.split_text(text)

    print(f"number of docs are {len(docs)}")
    return docs


def creat_embedding(docs):
    embedding = OpenAIEmbeddings()
    doc_search = FAISS.from_texts(docs, embedding)
    return doc_search

def responce_chain(doc_search, prompt, LLM):
    from langchain.chains.question_answering import load_qa_chain

    chain = load_qa_chain(llm=LLM, chain_type="stuff")

    docs = FAISS.similarity_search(doc_search, prompt)

    response = chain.run(question=prompt, input_documents=docs)

    return response


conversation_history = []

def ask(file_name, question):

    global conversation_history

    LLM = ChatOpenAI(
        temperature= 1.0,
        model = "gpt-3.5-turbo",
        openai_api_key = openai.api_key
    )

    
    prompt = "\n".join(conversation_history) + "\n" + question

    response = responce_chain(
        creat_embedding(creat_docs(load_and_extract_one_pdf(file_name))), prompt=prompt, LLM = LLM)

    
    #conversation_history.append(response)

    return response


    

