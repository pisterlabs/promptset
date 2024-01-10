import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import random
import time


st.set_page_config(
    page_title="Quiz",
    page_icon="👋",
)

# Title
st.markdown("<h2 style='color: green; font-style: italic; font-family: Comic Sans MS; ' >EcoKids Hub Quiz 🧠📝🏆</h2> <h3 style='color: #ADFF2F; font-style: italic; font-family: Comic Sans MS;'>Quiz with a Green Twist: Test Your Eco-Knowledge!</h3>", unsafe_allow_html=True) 

st.markdown("<p style='color: #4FC978; font-style: italic; font-family: Comic Sans MS; ' >Choose Your Eco-Adventure: Select a Topic, and LLM will quiz you on the fascinating world of sustainability based on the chosen-topic! </p>", unsafe_allow_html=True)

prompt = st.text_input('Enter the quiz-topic name here 🌱📝') 


# prompt = st.text_input('Plug in your prompt here') 

question_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Please generate sustainable environment related question for kids related to {topic} that requires a 1-word answer.'
)

correct_template = PromptTemplate(
    input_variables = ['question'], 
    template='In less than 3 words answer to {question}'
)

wrong_template = PromptTemplate(
    input_variables = ['question'], 
    template='In less than 3 words give wrong answer to {question}'
)

llm = OpenAI(temperature=0.9)

question_chain = LLMChain(llm=llm, prompt=question_template, output_key='question')
correct_chain = LLMChain(llm=llm, prompt=correct_template, output_key='correct')
wrong_chain = LLMChain(llm=llm, prompt=wrong_template,  output_key='wrong')


def answers(correct, wrong):
    num = random.randint(0, 1)
    list = []
    if num==0:
        list.append(correct)
        list.append(wrong)
    else:
        list.append(wrong)
        list.append(correct)
    return list,num

def cur(num, cur_num):
    if num==cur_num:
        return True
    return False

if prompt:
     question = question_chain.run(prompt)
     correct = correct_chain(question)
     wrong = wrong_chain(question)
     ans, num = answers(correct.get("correct"),wrong.get("wrong"))
     st.header(question) 
     st.write(ans[0])
     st.write(ans[1])
     st.write("")
     answer = st.text_input('Enter the quiz-topic answer here 🌱📝')
     
   
     if answer.strip().lower() == correct.get("correct").strip().lower() :
        st.write("Yeh you are right")
        st.balloons()

   

     

