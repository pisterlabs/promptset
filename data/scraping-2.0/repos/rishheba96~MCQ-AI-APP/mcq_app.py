# Import required modules
import os 
import json
import time
 
from langchain import  PromptTemplate

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI


import streamlit as st

def create_the_quiz_prompt_template():
    """Create the prompt template for the mcq app."""
    
    template = """
        You are an expert quiz maker. 
        create a quiz with {num_questions} Multiple-choice questions about the following topic: {quiz_topic}.

        The format of the output will be returned as: json object only with keys: Q1, O1, A1 without any other text
        <Q1>: Question1
        <O1>: <a. Answer 1>| <b. Answer 2>| <c. Answer 3>| <d. Answer 4>
        <A1>: <a|b|c|d>
        <Q2>: Question2
        <O2>: <a. Answer 1>| <b. Answer 2>| <c. Answer 3>| <d. Answer 4>
        <A2>: <a|b|c|d>

    """

    prompt = PromptTemplate.from_template(template)
    prompt.format(num_questions=2, quiz_topic=" ")

    return prompt

def map_letter_to_index(letter):
    """map each answer with its crossponding index  based on its Ascii code """
    letter = letter.lower() 
    if letter in {'a', 'b', 'c', 'd'}:
        return ord(letter) - ord('a')
        
def create_quiz_chain(prompt_template,llm):
    """Creates the chain for the quiz app."""
    return LLMChain(llm=llm, prompt=prompt_template)

def single_choice_fun(question, options, answer_index):
    """ creat the question and its choices, only one chice is corret"""
    user_answer = st.radio(question, options, index=None)

    if user_answer is None :  # if sol is empty 
        return False

    if options.index(user_answer) == answer_index:
        return True
    else:
        return False


# define st session vriable states 
if "output_questions" not in st.session_state:
    st.session_state["output_questions"] = ""  # 

if "json_obj_saved" not in st.session_state:
    st.session_state["json_obj_saved"] = ""

if "output_score" not in st.session_state:
    st.session_state["output_score"] = ""

if "num_correct_answer" not in st.session_state:
    st.session_state["num_correct_answer"] = 0

if "already_generated" not in st.session_state:
    st.session_state["already_generated"] = False

if "num_questions" not in st.session_state:
    st.session_state["num_questions"] = 1 

if "answers" not in st.session_state:
    st.session_state["answers"] = "" 


def main():
    # steup the main page of the application 
    st.title("AI Powered Quiz Generator")
    st.write("This app generates a quiz based on a given topic.")

    prompt_template = create_the_quiz_prompt_template()
    llm = ChatOpenAI()
    chain = create_quiz_chain(prompt_template,llm)
    tpoic = st.text_area("Enter the topic for the quiz")

    num_questions = st.number_input("Enter the number of questions",min_value=1,max_value=10,value=3)

    if st.button("Generate Quiz"):  # to generate the questions according to the inputs 
      
        st.session_state["num_questions"] = num_questions  # save number of question in the st seesion state 
        
        quiz_response = chain.run(num_questions=num_questions, quiz_topic=tpoic) # generate the questions based on the topic 

        # Convert the json output string  to a Python dictionary
        json_obj = json.loads(quiz_response)
        st.session_state["json_obj_saved"] = json_obj  # save the result((pyhton dict ) on session state 

        #

        num_correct_answer = 0
        st.session_state["num_correct_answer"] = 0  # to track number of correct answers 

        # Initialize lists to store extracted information
        questions, options, answers = [], [], []

        # Parse questions amd answres from the json obj 
        for key, value in json_obj.items():
            if key.startswith('Q'):
                questions.append(value)
            elif key.startswith('O'):
                options.append(value)
            elif key.startswith('A'):
                answers.append(value)
                st.session_state["answers"] += value

        # Print the extracted information
        for q, o, a in zip(questions, options, answers):

            correct_answer = single_choice_fun(q ,o.split('|'),map_letter_to_index(a))

            if correct_answer:
                num_correct_answer += 1
                st.session_state["num_correct_answer"] = num_correct_answer

        st.session_state["output_score"] = st.session_state["num_correct_answer"] / st.session_state["num_questions"]


        st.session_state["output_questions"] = "output"


    if st.session_state["output_questions"] != "" and st.session_state["already_generated"]:

        num_correct_answer = 0
        st.session_state["num_correct_answer"] = 0

        dict_json_obj_saved = st.session_state["json_obj_saved"]


        questions, options, answers = [], [], []

        for key, value in dict_json_obj_saved.items():

            if key.startswith('Q'):
                questions.append(value)
            elif key.startswith('O'):
                options.append(value)
            elif key.startswith('A'):
                answers.append(value)

        # Print the extracted information
        for q, o, a in zip(questions, options, answers):

            correct_answer = single_choice_fun(q ,o.split('|'),map_letter_to_index(a))

            if correct_answer:
                    num_correct_answer += 1
                    st.session_state["num_correct_answer"] = num_correct_answer

            st.session_state["output_score"] = st.session_state["num_correct_answer"] / st.session_state["num_questions"]

    if st.session_state["output_questions"] != "":
        st.session_state["already_generated"] = True


        # to generate the Answers 
        generate_button = st.button("Get results")

        if generate_button:
            if st.session_state["output_score"]:
                st.success(f'Reults is {st.session_state["output_score"]}', icon="✅")
            else:
                st.success(f'Reults is {st.session_state["output_score"]}', icon="✅")

            st.write(f"Answers:")

            for i in list(st.session_state['answers']):

                st.write(i)


if __name__=="__main__":
    main()





