import os
from apikey import apikey
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import streamlit as st
import pandas as pd

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey

# Set Streamlit app theme
st.markdown("""
    <style>
        body {
            background-color: #F8E1E7;  /* Light Pink background color */
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            color: #3E4E50;  /* Dark Gray text color */
        }
        .st-bw {
            color: #FF7E8D;  /* Rose color for emphasized text */
        }
        .stButton > button {
            background-color: #FF7E8D;  /* Rose color for buttons */
            color: #FFFFFF;  /* White text color for buttons */
        }
        .stTextInput > div > div > div > input {
            background-color: #FFD2D5;  /* Light Rose color for text input */
            color: #3E4E50;  /* Dark Gray text color for text input */
        }
        .stTextInput > div > div > div > label {
            color: #3E4E50;  /* Dark Gray text color for labels */
        }
        .stTextInput > div > div > div > div > svg {
            fill: #3E4E50;  /* Dark Gray fill color for icons */
        }
    </style>
    """, unsafe_allow_html=True)

# Title and image
st.title('Lucy-AI')

# Input prompt for the main chatbot
main_prompt = st.text_input('Ask me anything', help='Enter your question here')

# Previous inputs and responses
if 'previous_data' not in st.session_state:
    st.session_state.previous_data = []

# Save and display previous inputs and responses
if main_prompt:
    st.session_state.previous_data.append({"input": main_prompt, "response": ""})

# Display previous inputs and responses
for item in st.session_state.previous_data:
    st.write(f"Input: {item['input']}")
    st.write(f"Response: {item['response']}")
    st.write("---")

# Define prompt and template for the main chatbot
main_template = PromptTemplate(
    input_variables=['topic'],
    template='Chat with Lucy-AI about {topic}'
)

# LLM for the main chatbot
main_llm = ChatOpenAI(
    temperature=0.8,  # Adjusted temperature for more creative and varied responses
    model_name="gpt-3.5-turbo",
    frequency_penalty=0.5,
    max_tokens=500,
    streaming=True,
)
main_chain = LLMChain(llm=main_llm, prompt=main_template, verbose=True)

# Sidebar with additional features
st.sidebar.title("Choose Chat Section")

# Define prompts and templates for additional chatbots
prompt1 = st.sidebar.text_input('Enter medical term (Medical Terms)', help='Type the medical term you want to understand')
template1 = PromptTemplate(
    input_variables=['prompt'],
    template='Convert this medical language to layman language {prompt}'
)

prompt2 = st.sidebar.text_input('Enter mental health topic (Mental Health Advice)', help='Type the mental health topic')
template2 = PromptTemplate(
    input_variables=['topic'],
    template='Provide empathetic mental health advice for feeling {topic}'
)

prompt3 = st.sidebar.text_input('Enter movie genre and language (Movie Suggester)', help='Type the movie genre and language')
template3 = PromptTemplate(
    input_variables=['genre', 'language'],
    template='Suggest a movie in {genre} genre and {language} language'
)

# Period tracking
if st.sidebar.button('Period Tracking', key='btn_period', help='Track your menstrual cycle'):
    last_period_date = st.date_input('Enter the date of your last period', help='Select the date')
    cycle_length = st.number_input('Enter your cycle length (in days)', min_value=1, max_value=50, value=28,
                                   help='Choose the length of your menstrual cycle')

    # Predict the next period date
    next_period_date = last_period_date + pd.DateOffset(days=cycle_length)
    st.sidebar.write(f"Your estimated next period date is: {next_period_date.strftime('%Y-%m-%d')}")

# LLMs for additional chatbots
llm1 = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    frequency_penalty=0.5,
    max_tokens=500,
    streaming=True,
)
llm2 = ChatOpenAI(
    temperature=0.8,  # Adjusted temperature for more creative and varied responses
    model_name="gpt-3.5-turbo",
    frequency_penalty=0.5,
    max_tokens=500,
    streaming=True,
)
llm3 = ChatOpenAI(
    temperature=0.8,  # Adjusted temperature for more creative and varied responses
    model_name="gpt-3.5-turbo",
    frequency_penalty=0.5,
    max_tokens=500,
    streaming=True,
)

chain1 = LLMChain(llm=llm1, prompt=template1, verbose=True)
chain2 = LLMChain(llm=llm2, prompt=template2, verbose=True)
chain3 = LLMChain(llm=llm3, prompt=template3, verbose=True)

# Run additional chatbots
if prompt1:
    response1 = chain1.run(prompt=prompt1)
    st.sidebar.write(response1)

if prompt2:
    response2 = chain2.run(topic=prompt2)
    st.sidebar.write(response2)

if prompt3:
    response3 = chain3.run(genre=prompt3.split()[0], language=prompt3.split()[1])
    st.sidebar.write(response3)
