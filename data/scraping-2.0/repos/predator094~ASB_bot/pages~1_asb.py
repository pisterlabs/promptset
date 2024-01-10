import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import get_openai_callback
import PyPDF2
from docx import Document
from langchain.output_parsers import CommaSeparatedListOutputParser
from tools import *
import time
import os

api_key = st.secrets["OPENAI_API_KEY"]
# page setter
st.set_page_config(page_title="Assignment Asolver Bot", page_icon="📝")

# llm
llm = OpenAI(temperature=0, verbose=True, max_tokens=-1, api_key=api_key)
# agents
agent_executor = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106"),
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    max_execution_time=30,
)

# question template
question_template = PromptTemplate(
    input_variables=["prompt", "nop"],
    template="""Given the provided document, please extract all the questions and their sub-questions, correct any minor grammatical errors in the questions, and list each question with its subparts as individual elements in a Python list. Each question should end with a literal '##i,' where 'i' represents the question number. Ensure that the questions are properly partitioned and that each question, along with its subparts, forms a single list item. there should be only 5 list items,there should be only {nop} list items  .        

Document:
""{prompt}""

Output format:
a Python List

example :-
["question1...","questions2.....",....]
1. Read the provided document.
2. Identify and extract all the questions present in the document.
3. If you come across any minor grammatical errors in the questions, please correct them. 
4. string with proper partitioning of questions
5. Each question along with its sub parts should form a single list item
6. subparts of a question should not be entered as a different list item.
""",
)
title_chain = LLMChain(
    llm=ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-1106"),
    prompt=question_template,
    verbose=True,
    output_key="output",
)

q_template = PromptTemplate(
    input_variables=["prompt"],
    template="Answer the asked question precisely and briefly , if you dont know the answer just say it dont give hallucinated answers,if needed attach a link to any source needed for answering the question for example diagrams , blogposts ,updates, experiments ,etc\n Here's the question :- \n'{prompt}'",
)

# App framework

# sidebar
if "file" not in st.session_state:
    st.session_state.file = {0: False, 1: None, 2: [], 3: False}
with st.sidebar:
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            # For plain text files
            st.text("Text file content:")
            decoded_content = uploaded_file.read().decode("utf-8")
            st.session_state.file[1] = decoded_content
        elif uploaded_file.type == "application/pdf":
            # For PDF files
            st.text("PDF file content:")
            pdf = PyPDF2.PdfReader(uploaded_file)
            pdf_text = ""
            for page_num in range(len(pdf.pages)):
                pdf_text += pdf.pages[page_num].extract_text()
            st.session_state.file[1] = pdf_text
        elif (
            uploaded_file.type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            # For DOCX files
            st.text("DOCX file content:")
            doc = Document(uploaded_file)
            docx_text = ""
            for paragraph in doc.paragraphs:
                docx_text += paragraph.text
            st.session_state.file[1] = docx_text
        else:
            st.text("Unsupported file type")
        # byte_data = uploaded_file.read()
        # st.write(byte_data.decode("latin-1").encode("utf-8"))
        # loader = PyPDFLoader(f"../{uploaded_file.name}")

        # pages = loader.load_and_split()
        # ques = title_chain(pages)
        # questions = ques.split("\n")
        # print(questions)D
        if nop := st.text_input(
            "Enter no. of questions in the assignment", placeholder="Eg:- 5"
        ):
            st.session_state.file[0] = True
    else:
        st.session_state.file[0] = False
# main window
st.title("Assignment solver bot")
topph = st.empty()
# button keys
if "clicked" not in st.session_state:
    st.session_state.clicked = {1: False, 2: False}


def clicked(button):
    st.session_state.clicked[button] = True


# backend functions
def question_solver(qlist: list):
    for item in qlist:
        st.header(item)
        with st.spinner("Fetching the answer..."):
            answer = agent_executor.run(item)
            st.write(answer)
        st.success(f"Question {qlist.index(item)+1} completed ✅")
        st.toast(f"Solution to Q{qlist.index(item)+1} has been uploaded", icon="😍")
        time.sleep(20)
    st.toast("Your assignment is complete", icon="😊")


def question_list_maker():
    if st.session_state.clicked[1] == True:
        question_solver(st.session_state.file[2])
    elif st.session_state.clicked[2] == True or st.session_state.file[3] == False:
        st.title("Check questions :")
        placeholder = st.empty()
        ques = title_chain({"prompt": st.session_state.file[1], "nop": nop})
        # st.write(ques["output"])
        st.session_state.file[2] = eval(ques["output"])
        while placeholder.container():
            for items in st.session_state.file[2]:
                st.header(items)
            break
        st.session_state.clicked[2], st.session_state.file[3] == False, True
        correct_button = st.button(
            "Correct ✅", type="primary", key="my_button", on_click=clicked, args=[1]
        )
        retry_button = st.button(
            "Retry 🔃",
            type="secondary",
            key="my_button_retry",
            on_click=clicked,
            args=[2],
        )


# ren = []

# for items in st.session_state:
#     ren.append(str(items.items() if type(items) == "<class 'dict'>" else items))
#     if items == "file":
#         print(st.session_state.file)
# topph.write("\n".join(ren))
# print(ren)


def mainframe():
    with get_openai_callback() as cb:
        question_list_maker()

        st.session_state.file[0] = False
        print(cb)


# file solver
if st.session_state.file[0] == True:
    mainframe()
