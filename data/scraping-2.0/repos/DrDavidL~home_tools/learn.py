
import streamlit as st
import os 
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
from prompts import * 
from openai import OpenAI

def gen_response(messages, temperature, model, print = True):
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(
        api_key=api_key,
    )
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": print,
    }
    try:    
        completion = client.chat.completions.create(**params)
    except Exception as e:
        st.write(e)
        st.write(f'Here were the params: {params}')
        return None        
    with st.chat_message("assistant"):
        placeholder = st.empty()
    full_response = '' 
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            # full_response.append(chunk.choices[0].delta.content)
            placeholder.markdown(full_response)
    placeholder.markdown(full_response)
    return full_response


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            # del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.write("*Please contact David Liebovitz, MD if you need an updated password for access.*")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True  
    
def main():

    st.set_page_config(page_title='My Tutor', layout = 'centered', page_icon = ':stethoscope:', initial_sidebar_state = 'auto')
    st.title("Learn!")
    st.write("ALPHA version 0.5")


    with st.expander('Important Disclaimer'):
        st.write("Author: David Liebovitz")
        st.info(disclaimer)
        st.session_state.temp = st.slider("Select temperature (Higher values more creative but tangential and more error prone)", 0.0, 1.0, 0.3, 0.01)
        st.write("Last updated 12/9/23")
        


        



    if "current_thread" not in st.session_state:
        st.session_state["current_thread"] = ""


    if "last_answer" not in st.session_state:
        st.session_state["last_answer"] = []


    if "temp" not in st.session_state:
        st.session_state["temp"] = 0.3
        
    if "your_question" not in st.session_state:
        st.session_state["your_question"] = ""
        
    if "texts" not in st.session_state:
        st.session_state["texts"] = ""
        
    if "retriever" not in st.session_state:
        st.session_state["retriever"] = ""
        
    if "model" not in st.session_state:
        st.session_state["model"] = "openai/gpt-3.5-turbo-16k"
        
    if "tutor_user_topic" not in st.session_state:
        st.session_state["tutor_user_topic"] = []

    if "tutor_user_answer" not in st.session_state:
        st.session_state["tutor_user_answer"] = []
        
    if "message_thread" not in st.session_state:
        st.session_state["message_thread"] = []


    if check_password():
        

        
        embeddings = OpenAIEmbeddings()
        if "vectorstore" not in st.session_state:
            st.session_state["vectorstore"] = FAISS.load_local("bio.faiss", embeddings)
            


            
        model = st.sidebar.selectbox("Select a model", ["gpt-4-1106-preview", "gpt-3.5-turbo-1106", ])
        
        name = st.text_input("Please enter your first name:")
        if st.session_state.message_thread == []:
            st.warning("Enter your request at the bottom of the page.")
        user_input = st.chat_input("Your input goes here, ask to teach or for test questions, submit your responses, etc.:")    
        system_context = bio_tutor.format(name = name, outline = biology_outline)
        if st.session_state.message_thread == []:
            st.session_state.message_thread = [{"role": "system", "content": system_context}]
            

        if user_input:
            st.session_state.message_thread.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.spinner("Thinking..."): 
                answer_for_learner = gen_response(messages = st.session_state.message_thread, temperature = st.session_state.temp, model = model, print = True)
            st.session_state.tutor_user_topic.append(f'{name}: {user_input}')
            st.session_state.tutor_user_answer.append(answer_for_learner)        
            st.session_state.message_thread.append({"role": "assistant", "content": answer_for_learner})
            

            tutor_download_str = f"{disclaimer}\n\ntutor Questions and Answers:\n\n"
            for i in range(len(st.session_state.tutor_user_topic)):
                tutor_download_str += f"{st.session_state.tutor_user_topic[i]}\n"
                tutor_download_str += f"Answer: {st.session_state.tutor_user_answer[i]}\n\n"
                st.session_state.current_thread = tutor_download_str

            # Display the expander section with the full thread of questions and answers
            
        if st.session_state.message_thread != "":    
            with st.sidebar.expander("Your Conversation", expanded=False):
                for i in range(len(st.session_state.tutor_user_topic)):
                    st.info(f"{st.session_state.tutor_user_topic[i]}", icon="🧐")
                    st.success(f"Answer: {st.session_state.tutor_user_answer[i]}", icon="🤖")

                if st.session_state.current_thread != '':
                    st.download_button('Download', st.session_state.current_thread, key='tutor_questions')
        
        if st.sidebar.button("Start a new conversation"):
            st.session_state.message_thread = []
            
if __name__ == "__main__":
    main()