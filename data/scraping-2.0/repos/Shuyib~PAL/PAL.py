""" Program Aided Language model: Helps with math problems 
    This example was adapted from Data Professor Github repo 
    It uses streamlit to create a web app that uses OpenAI API to generate responses to math questions
    Langchain is used to chain the responses to create a coherent response using PALChain module
"""
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import PALChain
from langchain.chains.llm import LLMChain

# set page config
# Add title and description
st.set_page_config(page_title="🦜🔗 Program Aided Language model")
st.title("🦜🔗🧮 Program Aided Language Model: Helps with math problems")
st.markdown(
    """This example was adapted from Data Professor Github [repo](https://github.com/dataprofessor/langchain-quickstart/blob/master/streamlit_app.py)"""
)
st.markdown(
    """Paper: [Program-Aided Language Models for Program Synthesis](https://arxiv.org/pdf/2211.10435.pdf)"""
)
st.markdown(
    """Credit: [Sam Witteven](https://www.youtube.com/playlist?list=PL8motc6AQftk1Bs42EW45kwYbyJ4jOdiZ)"""
)

# sidebar for OpenAI API key & model selection
openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
select_instruct_model = st.sidebar.selectbox(
    "Select Instruction Model",
    ("text-davinci-003", "gpt-3.5-turbo-instruct"),
)


def generate_response(input_text):
    """
    Generates response to input text using PALChain

    Parameters
    ----------
    input_text : str
      Input text to generate response for using PALChain

    Returns
    -------
    None

    Example
    -------
    >>> generate_response("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?")
    """
    llm = OpenAI(
        temperature=0,
        openai_api_key=openai_api_key,
        max_tokens=512,
        model=select_instruct_model,
    )
    pal_chain = PALChain.from_math_prompt(llm, verbose=True)
    st.markdown(pal_chain.run(input_text))


# Start a new form named "my_form"
with st.form("my_form"):
    # Create a text area for the user to input a math question. The text area is pre-filled with a default question.
    text = st.text_area(
        "Enter math question:",
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
    )
    # Create a submit button for the form. When the button is clicked, the form is submitted and the page reruns from the top.
    submitted = st.form_submit_button("Submit")
    # Check if the OpenAI API key is not valid (it should start with "sk-"). If it's not valid, display a warning message.
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    # If the form is submitted and the OpenAI API key is valid, generate a response to the user's question.
    if submitted and openai_api_key.startswith("sk-"):
        generate_response(text)
