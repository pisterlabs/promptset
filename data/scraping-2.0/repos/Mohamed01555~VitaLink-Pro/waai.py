# import gpt3
from utils import *
import base64
from time import sleep
from asyncio import run
from langchain.prompts import PromptTemplate
import streamlit as st
from pathlib import Path
from streamlit_option_menu import option_menu
import g4f

question_prompt_template = """
            Role: Personalized Virtual Health Assistant (PVHA)

            Main Task: Act like a physical fitness trainer with 30 years of experience and
                       provide actionable health advice and recommendations based on the my profile.

            My Profile:
            - Prefered language of answer: {lang}
            - Current Fitness Level: {cfl} (0 is the lowest, 10 is the heighest)
            - Health Goals: {hg}
            - vergitables and fruits available to me to suggest healthy meals suitable to my goal are {vf}
            - Protein Supplement Use: {psu}
            - Chronic Diseases/Health Conditions: {cd}
            - Physical Disabilities/Limitations: {pd}
            - Age: {age} years
            - My Free Time for Physical Activity: {ft}
            - Gender: {g}
            - pregnance status if Female: {ps}
            - you may take other parameters. Take them also into considration while generating the response.

            My query is : {q}

            Your previous answer is : {prev_answer}

            Based on my profile provide an answer to my query with a plan to implement it.
            
            note that my query may be thanking for your previous response, please reply with a suitable answer.
            
            Constraints: You must must must provide the answer my Prefered language

            Additionally, provide your answer with links to resources that helped u answering my question.
            
            Please don't meantion your Identity in the answer,e.g. don't say I am GPT 3.5 or GPT4 or whatever u r, instead act like a physical fitness trainer.
            
            Your answer in bullet points:
        """

prompt = PromptTemplate(input_variables=['lang', "cfl","hg", 'vf', "psu", "cd", "pd", "age", "ft", "g", "ps", 'q', 'prev_answer'], template=question_prompt_template)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def main():
    # setup streamlit page
    st.set_page_config(
        page_title="VitaLink Pro",
        page_icon="logo.jpeg")
    
    option = option_menu(
    menu_title=None,
    options=["Home", "FAQs", "Contact"],
    icons=["house-check", "patch-question-fill", "envelope"],
    orientation='horizontal',
    styles={
        "container": {"padding": "0!important", "background-color": "#333"},        
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#ff9900"},
        "nav-link-selected": {"background-color": "#6c757d"},
    }
    )   

    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown(html_code, unsafe_allow_html=True)

    # initialize responses.
    if "responses" not in st.session_state:
        st.session_state.responses = []
    
    if "question" not in st.session_state:
        st.session_state.question = None

    with st.sidebar:
        title = st.markdown("""**Hello, Welcome to `Vitalink Pro!`. I can help you in:
                            <ul>
                                <li>Guiding you to increase the effectiveness of the your physical activity.</li>
                                <li>Recommending types of sports and activities suitable to you.</li>
                                <li>Suggesting healthy food based on your available food and goal.</li>
                                <li>Suggesting ways to incorporate physical activity into the your daily routine.</li>
                                <li>Outlining the amount of physical activity suitable to you.</li>
                                <li>Identifing Aerobic sports suitable to you.</li>
                                <li>Offering general tips for a healthy lifestyle.</li>
                                <li>And many more...</li>
                            </ul>
                                To provide you with a personalized health and fitness experience, we'd like to gather some information.
                                Please take a few minutes to answer the following questions:**""", unsafe_allow_html=True)
        
        language = st.radio('**What is your prefered language?**', ['العربية','English'])
        
        selected_value = st.slider("**Please, rate your current fitness level**", min_value=0, max_value=10, value=5, step=1)
        
        # Define a list of health goal options
        health_goal_options = ["Weight Loss", "Muscle Gain", "Overall Well-being"]

        # Display a multiselect for users to choose multiple health goals
        selected_health_goals = st.multiselect("**What are your primary health and fitness goals?**", health_goal_options)

        # Define a list of available vergitables and fruits
        veg_fru = st.text_input('**Please, mention fodd, vergitables,and fruits available to you to suggest a meal suitable to your goal.**')
        
        selected_protein_supplements = st.radio('**Are you taking Protein supplements?**', ['No','Yes'])
        
        selected_chronic_disease = st.radio('**Do you suffer from chronic diseases?**',  ['No','Yes'])
        if selected_chronic_disease == 'Yes':
            selected_chronic_disease = st.text_input('**Please, mention your chronic diseases**')
        
        selected_disability = st.radio('**Do you suffer from a disability?**', ['No','Yes'])
        if selected_disability == 'Yes':
            selected_disability = st.text_input('**Please, mention your disability.**')

        age = st.number_input('**Enter your age, please.**')

        free_time = st.text_input('**When is your free time?**')

        gender = st.radio('**Please, Enter your gender.**', ['Male', 'Female'])
        
        is_pregnant = None
        if gender == 'Female':
            is_pregnant = st.radio('**Are you pregnant?**', ['I am pregnant', 'I am not pregnant'])

    if option == 'Home':
        for response in st.session_state.responses:
            with st.chat_message(response['role']):
                st.markdown(response['content'], unsafe_allow_html=True)
        
        st.session_state.question = st.chat_input('Message me...', key = 'giving a question')
        if st.session_state.question:
            with st.chat_message('user'):
                st.markdown(st.session_state.question, unsafe_allow_html=True)

            st.session_state.responses.append({'role':"user", 'content': st.session_state.question})
            with st.spinner("Please, don't enter a new question or change anything in the sidebar while the answer is being generated!"):
                with st.chat_message('assistant'):
                    st.session_state.message_placeholder = st.empty()

                    query = prompt.format(lang = language, cfl = selected_value, hg = selected_health_goals, vf = veg_fru, psu = selected_protein_supplements, cd = selected_chronic_disease,
                                        pd = selected_disability, age = age, ft = free_time, g = gender, ps = is_pregnant,
                                        q = st.session_state.question, prev_answer = st.session_state.responses[-2]['content']if len(st.session_state.responses) >= 2 else '')   

                    response = g4f.ChatCompletion.create(model=g4f.models.gpt_35_turbo_16k_0613, messages=[{"role": "user", "content": query}], stream=True)  # Alternative model setting          
                    res = ''
                    for r in response:
                        res += r
                        st.session_state.message_placeholder.markdown(res, unsafe_allow_html=True)                   
            
                st.session_state.responses.append({'role' : 'assistant', 'content' : res})
           
    elif option == 'FAQs':
        FAQs()
    elif option == 'Contact':
        contact()
    else:
        donate()

if __name__ == '__main__':
    main()
