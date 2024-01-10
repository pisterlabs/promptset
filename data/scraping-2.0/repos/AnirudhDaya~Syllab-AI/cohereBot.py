import cohere
from dotenv import load_dotenv
import os
import streamlit as st
import extraction
import re
import json
import database as db
# os.environ['COHERE_API_KEY'] = os.getenv("COHERE_API_KEY")
co = cohere.Client(st.secrets["COHERE_API_KEY"]) 

def cohereCall(syllabus,questions_set,not_exists):

    all_responses = ""
    st.write("LLM RUNNING ... ")

    # questions_set,not_exists,collectionNameRaw,all_questions = extraction.extract()

    if (not_exists):

        print("EXtracting SYLLABUS...")
        # syllabus = extraction.extract_syllabus()
        preamble_prompt = """ You are an expert in finding out the most relevant topics based on the following syllabus : """ + syllabus

        for i in range(0,len(questions_set)):
            questions = questions_set[i]
            message_prompt = """I am going to give you a set of questions.
            You already have the syllabus. So I want to you to categorize each 
            question based on what topic in the syllabus it is most related to. 
            (If a question is realted to more than one topic included it in 
            all of them) and rank the topics based on the numbers of questions 
            under each topic and display the top 10 rankings only nothing 
            else.
            """ + questions.replace("\n","")

            response = ""

            if type == "condense" :
                print("Calling COHERE CHat")
                response = co.chat(
                    message=message_prompt,
                    # documents=[],
                    model='command',
                    temperature=0.2,
                    # return_prompt=True,
                    preamble_override=preamble_prompt,
                    chat_history=[],
                    connectors=[{"id": "web-search"}],
                    prompt_truncation='auto',
                )
                st.write(response.text)
            all_responses +=" "+response.text


    pattern = r"([a-zA-Z]+)\.pdf$"
    match = re.search(pattern, collectionNameRaw)

    if(type == "condense"):
        if match:
            collectionName=match.group(1)
            db.db_questions(all_responses,collectionName)
            # db.TopTenTopics()
            st.write("Top 10 topics are: \n",db.TopTenTopics(collectionName))
            st.write("____PREV CALL #1 (CONDENSE) AREA _____ END")

    elif(type == "question"):

        st.write("___ COHERE CALL #2 (QUESTION) AREA START_____ ")
        cleaned_questions = extraction.extract_ktu_questions(all_questions[0])
        print("Cleaned Questions : ",cleaned_questions)

        # Print the extracted questions as JSON
        questions_json = json.dumps(cleaned_questions, indent=2)

        # Check if questions is not empty before proceeding
        if questions_json:
            response = co.chat(
                message=query,
                model='command',
                temperature=0.2,
                documents=cleaned_questions,
                preamble_override="You are an expert in creating question paper based on the following syllabus",
                # return_prompt=True,
                chat_history=[],
                # connectors=[{"id": "web-search"}],
                prompt_truncation='auto',
            )
        else:
            st.write("Questions not found")

        st.write(response.text)

        # print("CAlling QUESTION COHERE CHat")

def demoCall():
    print("IN COMPONENT")
    st.write("IN  COHERE BOT FILE")


def main():
    cohereCall("Who is ELon musk ?")


if __name__== "__main__":
    main()
