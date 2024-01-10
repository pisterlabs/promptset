import streamlit as st
import matplotlib.pyplot as plt
import io
import base64
import pickle
from PIL import Image

from utils import *
import constants


import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

def load_data():
  """Loads data from the 'data.pickle' file."""

  topic_info = ""
  prob_tree_fig = None
  topic_challenges = ""
  sol_tree_plot = None
  sol_grades = None

  with open('data.pickle', 'rb') as file:
    top = pickle.load(file)
    topic_info = "Topic is " + top.name + "\n"
    prob_str = top.get_problems_str()
    prob_tree_fig = top.plot_hierarchy_problems()
    s1 = "\nfrom that problem we created a developed challenge. \n"
    topic_challenges = s1 + top.get_challenges_str()
    s2 = "\nFrom that challenge, we create 3 optional solutions:\n"
    sol_str = s2 + top.challenges[0].get_solutions_str()

    sol_tree_plot = top.challenges[0].plot_hierarchy_solutions((10, 4))
    sol_grades = plot_solutions_polygons(top.challenges[0].solutions)

  return topic_info, prob_str, prob_tree_fig, topic_challenges, sol_str, sol_tree_plot, sol_grades, top

def main():
    # Load data from your 'data.pickle' file
    topic_info, prob_str, prob_tree_fig, topic_challenges, sol_str, sol_tree_plot, sol_grades, top = load_data()
    # topic_info, prob_str, prob_tree_fig = load_data()

    # Display the topic infoz
    st.markdown(f"## {topic_info}")
    
    st.markdown(f"### The problems are:")
    st.pyplot(prob_tree_fig)

    st.write(prob_str)

    st.write("We look at 1 problem in praticular and analyze it")
    prob = top.problems[1]
    st.markdown(f"### {prob.sub_class}")
    st.pyplot(prob.build_knowledge_graph())
    
    # Display the challenges
    st.markdown(f"### Challenge")
    st.markdown(topic_challenges)

    # Display the solutions tree
    st.markdown(f"### Solutions Tree")
    st.pyplot(sol_tree_plot)
    st.markdown(f"### Solution Grades")
    st.pyplot(sol_grades)

    # Display the solution grades
    # st.write(sol_str)
    sols = top.challenges[0].solutions
    for i, sol in enumerate(sols):
        st.markdown(f"### {sol.sub_class}")
        st.write(sol.description)
        sol_input = st.text_input(f"solution_{i+1} update information", key=f"sol_input_{i}")
        if sol_input:
            sol.update_solution(sol_input)
            with open("streamlit_pkl.pickle", "wb") as file:
                file.write(pickle.dumps(top))
    
    chatbot_main2()


# Chat bot code

def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # # setup streamlit page
    # st.set_page_config(
    #     page_title="Connversetion with AI-Agent",
    #     page_icon="🤖"
    # )


def chatbot_main():
    init()

    # topic = "Backpack"
    top = None
    with open('data.pickle', 'rb') as file:
        top = pickle.load(file)
    problems = top.get_problems_str()
    challenges = top.get_challenges_str()
    solutions = top.challenges[0].get_solutions_str()
    chat = ChatOpenAI(temperature=0)

    # initialize message history
    if "messages" not in st.session_state:
        system_msg = f"""We are a company that makes {top.name} , we want to upgrade our product. 
        For that end we would like you to help our imployes understand and analyze the problems with the product and the solutions for those problems.
        For now our problems are: {problems}
        The Challenges are: {challenges}
        The Solutions are: {solutions}
        """
        st.session_state.messages = [
            SystemMessage(content=system_msg)
        ]

    st.header("discussion with AI-Bot🤖")

    # sidebar with user input
    with st.sidebar:
        user_input = st.text_input("Your message: ", key="user_input")

        # handle user input
        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking..."):
                response = chat(st.session_state.messages)
            st.session_state.messages.append(
                AIMessage(content=response.content))
    
    
    # display message history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')


def chatbot_main2():
    st.title("aristo-chatbot")

    # Set OpenAI API key from Streamlit secrets
    openai.api_key = constants.OPENAI_API_KEY

    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    # Initialize chat history
    
    # if "messages" not in st.session_state:
    #     st.session_state.messages = [{"role": "system", "content":"you are a chat bot with the name Chubby, and you finish each sentence with hoof!"}]
    top = None
    with open('data.pickle', 'rb') as file:
        top = pickle.load(file)
    problems = top.get_problems_str()
    challenges = top.get_challenges_str()
    solutions = top.challenges[0].get_solutions_str()

    # initialize message history
    if "messages" not in st.session_state:
        system_msg = f"""We are a company that makes {top.name} , we want to upgrade our product. 
        For that end we would like you to help our imployes understand and analyze the problems with the product and the solutions for those problems.
        For now our problems are: {problems}
        The Challenges are: {challenges}
        The Solutions are: {solutions}
        """
        st.session_state.messages = [{"role": "system", "content":system_msg}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
            ):  
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # if st.button("Save Chat history"):
    #     save_chat_history(st.session_state.messages, top)
    
    
if __name__ == '__main__':
  main()
