import streamlit as st
import matplotlib.pyplot as plt
import io
import base64
import pickle
from PIL import Image

from utils import *
from utils import Topic
import constants
import google_sheet as sheets


import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage)


def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")
    


def main():
    init()
    st.title("Aristo - The AI Assistant")
    topic_name = "Backpacks"
    st.markdown(f"## Our goal is to create a better {topic_name}")
    filesize = os.path.getsize("demo.pickle")
    if filesize == 0:
        create_from_zero(topic_name)

    else:
        with open("demo.pickle", "rb") as file:
            top = pickle.load(file)
            print("loaded pickle")
            continue_from_pickle(top)

def create_from_zero(topic_name):
    st.write("We colected different problems from people using google forms, and now we will analyze them:")
    responses = sheets.get_people_responses()
    problems = [resp['problem'] for resp in responses]
    top = Topic(topic_name)
    top.classify_problems(", ".join(problems))
    
    problems_tree = top.plot_hierarchy_problems()
    st.pyplot(problems_tree)
    st.write(top.get_problems_str())
    
    st.write("We look at 1 problem in praticular and analyze it")
    prob = top.problems[1]
    st.markdown(f"### {prob.sub_class}")
    # prob.create_factors()
    # prob_kg = prob.build_knowledge_graph()
    # st.pyplot(prob_kg)
    
    problems_to_chall = [0]
    top.create_challenge(problem_indexes=problems_to_chall)
    st.write(top.get_challenges_str())
    top.challenges[0].create_solutions(3)
    st.pyplot(top.challenges[0].plot_hierarchy_solutions())
    for sol in top.challenges[0].solutions:
        st.write(sol.sub_class + ": " + sol.description)
    st.pyplot(plot_solutions_polygons(top.challenges[0].solutions))
    
    # save the data to a pickle file
    save_button = st.button("Save data")
    if save_button:
        with open("demo.pickle", "wb") as file:
            pickle.dump(top, file)
    aristo_bot(top)
    
def continue_from_pickle(top):
    st.write("We collected different problems from people using google forms, and now we will analyze them:")
    st.markdown(f"## Problems Tree")
    
    problems_tree = top.plot_hierarchy_problems()
    st.pyplot(problems_tree)
    st.markdown("## Problems:")
    for prob in top.problems:
        st.markdown(f"### {prob.sub_class}")
        st.write(prob.description)
        prob_kg = prob.build_knowledge_graph()
        st.pyplot(prob_kg)
    
    # # problems_to_chall = [0]
    # # top.create_challenge(problem_indexes=problems_to_chall)
    # st.write(top.get_challenges_str())
    # # top.challenges[0].create_solutions(3)
    st.markdown("## Solutions Section")
    st.pyplot(top.plot_hierarchy_solutions())
    for sol in top.solutions:
        st.write(sol.sub_class + ": " + sol.description)
    st.pyplot(plot_solutions_polygons(top.solutions[:3], to_show=False))
    # aristo_bot(top)
    
    
def aristo_bot(top):
    # initialize message history
    st.title("aristo-chatbot")
    problems = top.get_problems_str()
    challenges = top.get_challenges_str()
    solutions = top.challenges[0].get_solutions_str()
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
        
# def load_data(file_path):
#   """Loads data from the 'data.pickle' file."""

#   topic_info = ""
#   prob_tree_fig = None
#   topic_challenges = ""
#   sol_tree_plot = None
#   sol_grades = None

#   with open(file_path, 'rb') as file:
#     top = pickle.load(file)
#     topic_info = "Topic is " + top.name + "\n"
#     prob_str = top.get_problems_str()
#     prob_tree_fig = top.plot_hierarchy_problems()
#     s1 = "\nWe look at 1 problem in particular and create from it a developed challenge. \n"
#     if top.challenges == []:
#         top.create_challenge()
#         print("created challenge")
#     topic_challenges = s1 + top.get_challenges_str()
#     if top.challenges[0].solutions == []:
#         top.challenges[0].create_solutions(3)
#     s2 = "\nFrom that challenge, we create 3 optional solutions:\n"
#     sol_str = s2 + top.challenges[0].get_solutions_str()

#     sol_tree_plot = top.challenges[0].plot_hierarchy_solutions((10, 4))
#     sol_grades = top.challenges[0].plot_solutions_polygons(to_show=False)

#   return topic_info, prob_str, prob_tree_fig, topic_challenges, sol_str, sol_tree_plot, sol_grades, top


# def main():
#     st.title("Aristo - The AI Assistant")
#     topic_name = "Backpacks"
#     st.markdown(f"## Our goal is to create a better {topic_name}")

#     st.markdown("### Problems Section")
#     st.write("Where should I get the problems from?")
#     # Create buttons
    
#     button1 = st.button("The google sheets file")
#     button2 = st.button("Generate random problems")
#     button3 = st.button("Use example")

#     # Check which button is clicked and show corresponding content
#     if button1:
#         responses = sheets.get_people_responses()
#         problems = [resp['problem'] for resp in responses]
#         top = Topic(topic_name)
#         top.classify_problems(", ".join(problems))
        
#         show_everything(top)

#     elif button2:           
#         top = Topic(topic_name)
#         top.generate_problems(3)
#         show_everything(top)
#         # with open("streamlit_pkl.pickle", "wb") as file:
#         #     pickle.dump(top, file)
    
#     elif button3:
#         top = None
#         with open('data.pickle', 'rb') as file:
#             top = pickle.load(file)
#         show_everything(top)
    
    
if __name__ == "__main__":
    main()
