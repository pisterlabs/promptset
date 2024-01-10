import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.document_loaders import *
from langchain.chains.summarize import load_summarize_chain
import tempfile
from langchain.docstore.document import Document
import os
from demogpt.plan.model import DemoGPT
from demogpt.plan.utils import runStreamlit
import webbrowser
import subprocess

# Initialize a variable to store the generated essay.
generated_essay = ""

# Functions for rephrasing
def textRephraser(original_text, model_choice, mode, keywords=[]):
    chat = ChatOpenAI(
        model=model_choice,
        temperature=0.7 if mode == 'Creative mode' else 0.2
    )

    system_template = "You are an assistant designed to rephrase the given text."

    if mode == 'Creative mode' and keywords:
        system_template += f" Incorporate these keywords: {', '.join(keywords)}."

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = f"Please rephrase the following text: '{original_text}'."
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(original_text=original_text)
    return result

# Function for generating essay
def generateEssay(prompt, model_choice):
    chat = ChatOpenAI(
        model=model_choice,
        temperature=0.7
    )

    system_template = f"You are an assistant designed to generate an essay on the given prompt."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = f"Please write an essay on: '{prompt}'."
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    essay = chain.run(original_text=prompt)
    return essay

# Define the generate_response function for the DemoGPT tool
def generate_response(txt, title):
    """
    Generate response using the LangChainCoder.

    Args:
        txt (str): The input text.

    Yields:
        dict: A dictionary containing response information.
    """
    for data in agent(txt, title):
        yield data

# Define the progress bar function
def progressBar(percentage, bar=None):
    if bar:
        bar.progress(percentage)
    else:
        return st.progress(percentage)

# Sidebar navigation
st.sidebar.title('GPTHero Navigation')
page = st.sidebar.selectbox('Choose a page:', ['Home', 'Generate Essay', 'Build an Application'])

# Page Routing for Home
if page == 'Home':
    st.title('GPTHero Rephraser')

    original_text = st.text_area("Enter the original text here")
    model_choice = st.selectbox("Select Model:", ["gpt-3.5-turbo", "gpt-4"])
    rephrase_mode = st.radio("Rephrasing Mode:", ["Conservative mode", "Creative mode"])
    keywords = []
    if rephrase_mode == "Creative mode":
        keywords = st.text_input("Enter keywords for rephrasing (comma separated):").split(',')

    if st.button('Rephrase Text'):
        if original_text:
            with st.spinner('Rephrasing...'):
                rephrased_text = textRephraser(original_text, model_choice, rephrase_mode, keywords)
            st.markdown(f"**Rephrased Text:** {rephrased_text}")
        else:
            st.warning('Please enter the original text to rephrase.')

# Page Routing for Generate Essay
elif page == 'Generate Essay':
    st.title('GPTHero Essay Generator')

    essay_prompt = st.text_area("Enter your essay prompt:")
    model_choice = st.selectbox("Select Model for Essay:", ["gpt-3.5-turbo", "gpt-4"])

    if st.button('Generate Essay'):
        with st.spinner('Generating Essay...'):
            generated_essay = generateEssay(essay_prompt, model_choice)
        st.text_area("Generated Essay:", value=generated_essay, height=400)

    edit_prompt = st.text_input("Provide a prompt to guide the creation of a new, similar essay based on the previous one:")
    if st.button('Create Similar Essay'):
        with st.spinner('Generating Similar Essay...'):
            generated_essay = generateEssay(edit_prompt, model_choice)
        st.text_area("New Similar Essay:", value=generated_essay, height=400)

# Page Routing for Build an Application
elif page == 'Build an Application':
    st.title('Build Your Own Application')

    st.write("""
    This feature allows you to easily build applications without needing to write code. 
    Specifically, for writing-based tasks, you can generate text and then rephrase it using GPTHero. 
    Simply select an example or describe your idea, and the system will generate the code for you.
    """)

    # Initialize session state variables
    if "done" not in st.session_state:
        st.session_state.done = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "code" not in st.session_state:
        st.session_state.code = ""
    if "edit_mode" not in st.session_state:
        st.session_state.edit_mode = False
    if "message" not in st.session_state:
        st.session_state.message = ""
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "site_deployed" not in st.session_state:
        st.session_state.site_deployed = False

    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        placeholder="sk-...",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
    )

    models = ("gpt-3.5-turbo-16k", "gpt-4")
    model_name = st.sidebar.selectbox("Model", models)

    # Predefined examples dropdown
    examples = ["Select an example", "Chatbot", "Story Generator", "Q&A System", "Blog Post Maker"]
    selected_example = st.selectbox("Choose a predefined example:", examples)

    demo_idea = ""
    demo_title = ""
    if selected_example == "Chatbot":
        demo_idea = "I want to build a chatbot that can answer questions about space."
        demo_title = "Space Chatbot"
    elif selected_example == "Story Generator":
        demo_idea = "I want an application that can generate short stories based on a prompt."
        demo_title = "Story Generator"
    elif selected_example == "Q&A System":
        demo_idea = "I want a system where I can input a question and get a detailed answer."
        demo_title = "Q&A System"
    elif selected_example == "Blog Post Maker":
        demo_idea = "I want an application that can generate blog posts based on a given topic."
        demo_title = "Blog Post Maker"
    else:
        demo_idea = st.text_area(
            "Enter your LLM-based demo idea", placeholder="Type your demo idea here", height=100
        )
        demo_title = st.text_input(
            "Give a name for your application", placeholder="Title"
        )

    agent = DemoGPT(openai_api_key=openai_api_key)
    agent.setModel(model_name)

    # Function to update the progress bar
    def progressBar(value, bar):
        bar.progress(value)

    # Form submission logic
    with st.form("a", clear_on_submit=True):
        submitted = st.form_submit_button("Generate Code")

    if submitted:
        if not openai_api_key.startswith("sk-"):
            st.warning("Please enter your OpenAI API Key!")
        else:
            bar = st.progress(0)
            try:
                for data in generate_response(demo_idea, demo_title):
                    done = data.get("done", False)
                    message = data.get("message", "")
                    stage = data.get("stage", "stage")
                    code = data.get("code", "")
                    progressBar(data["percentage"], bar)

                    st.session_state.done = True

                    if done:
                        st.session_state.code = code
                        st.success("Final code has been generated. Directing to the demo page...")
                        break

                    st.info(message)

                    st.session_state.messages.append(message)
            except KeyError as e:
                if str(e) == "'validate_input'":
                    st.error("The provided input could not be processed. Please refine your query.")
                else:
                    st.error(f"An error occurred: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif "messages" in st.session_state:
        for message in st.session_state.messages:
            st.info(message)

    if st.session_state.done:
        with st.expander("Code", expanded=True):
            code_empty = st.empty()
            if st.session_state.edit_mode:
                new_code = code_empty.text_area("", st.session_state.code, height=500)
                if st.button("Save & Rerun"):
                    st.session_state.code = new_code  # Save the edited code to session state
                    st.session_state.edit_mode = False  # Exit edit mode
                    code_empty.code(new_code)
                    st.experimental_rerun()

            else:
                code_empty.code(st.session_state.code)
                if st.button("Edit"):
                    st.session_state.edit_mode = True  # Enter edit mode
                    st.experimental_rerun()

        # Automatically deploy the application
        # Save the generated code to a temporary file
        with open("generated_app.py", "w") as f:
            f.write(st.session_state.code)
        # Start a new Streamlit app using the generated code on port 4000
        if not st.session_state.site_deployed:
            subprocess.Popen(["streamlit", "run", "generated_app.py", "--server.port", "4000"])
            webbrowser.open("http://localhost:4000")  # Automatically open the new app in a browser
            st.session_state.site_deployed = True

        # Allow the user to continuously revise the generated application by adding more queries
        user_query = st.text_input("Enter a query to edit the site:")
        if user_query and st.button("Apply Changes"):
            st.session_state.query = user_query
            # Regenerate the application based on the user's query
            bar = st.progress(0)
            try:
                for data in generate_response(user_query, demo_title):
                    done = data.get("done", False)
                    message = data.get("message", "")
                    stage = data.get("stage", "stage")
                    code = data.get("code", "")
                    progressBar(data["percentage"], bar)

                    st.session_state.done = True

                    if done:
                        st.session_state.code = code
                        st.success("Final code has been generated. Directing to the demo page...")
                        break

                    st.info(message)

                    st.session_state.messages.append(message)
            except KeyError as e:
                if str(e) == "'validate_input'":
                    st.error("The provided input could not be processed. Please refine your query.")
                else:
                    st.error(f"An error occurred: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

            # Save the regenerated code to a temporary file
            with open("generated_app.py", "w") as f:
                f.write(st.session_state.code)
            # Restart the Streamlit app using the regenerated code on port 4000
            subprocess.Popen(["streamlit", "run", "generated_app.py", "--server.port", "4000"])
            webbrowser.open("http://localhost:4000")  # Automatically open the new app in a browser