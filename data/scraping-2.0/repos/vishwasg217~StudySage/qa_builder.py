
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA


import streamlit as st
from pydantic import BaseModel, Field, validator
from typing import List

from utils import process_pdf

llm = None
db = None


template = """
You have the task of generating answers for the following questions from the notes given.
The answers should be explained in a simple manner with examples.


Here are the questions you need to generate answers for:

==================
Questions: {questions}

==================

{output_format_instructions}

"""

# creating a Pydantic model to parse the output
class GenerateAnswers(BaseModel):
    question: str = Field(description="Questions")
    answer: List[str] = Field(description="Answers")

    # @validator('summary', allow_reuse=True)
    # def has_three_or_more_lines(cls, list_of_lines):
    #     if len(list_of_lines) < 3:
    #         raise ValueError("Generated summary has less than three bullet points!")
    #     return list_of_lines

parser = PydanticOutputParser(pydantic_object=GenerateAnswers)  

prompt = PromptTemplate(template=template, 
                        input_variables=['questions'],
                        partial_variables={"output_format_instructions": parser.get_format_instructions()},  # used to format the output
)

def database(splitted_text):
    st.write("db")
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.API)
    db = Chroma.from_texts(splitted_text, embeddings)
    return db

def format_questions(questions):
    st.write("hello")
    llm = OpenAI(openai_api_key=st.session_state.API, temperature=0.5)
    formatted_questions = llm(f"split the given text into questions, do nothing else: {questions}")
    return formatted_questions
    


if __name__ == "__main__":

    st.session_state.API = st.sidebar.text_input("OpenAI API Key", key="OPEN_AI_API")

    if st.session_state.API:

        if "notes" not in st.session_state:
            st.session_state.notes = False


        llm = OpenAI(openai_api_key=st.session_state.API, temperature=0.5)
        notes = st.sidebar.file_uploader("Upload your notes", type=["pdf"])
        if st.sidebar.button("Process PDF"):
            st.session_state.notes = True
            with st.spinner("Processing PDF..."):
                splitted_text = process_pdf([notes])
                db = database(splitted_text)
            
            retriever = VectorStoreRetriever(vectorstore=db)
            st.session_state.qa_chain = RetrievalQA(llm=llm, retriever=retriever)


        if st.session_state.notes:
            questions = st.text_area("Enter your questions here", key="question")
            if questions:
                st.write("hello")
                formatted_questions = format_questions(questions)
                st.write("questions: ",formatted_questions)
                formatted_prompt = prompt.format_prompt(questions=formatted_questions)
                messages = [HumanMessage(content=formatted_prompt.to_string())]
                response = st.session_state.qa_chain({"query": messages})

                st.write(response)

            