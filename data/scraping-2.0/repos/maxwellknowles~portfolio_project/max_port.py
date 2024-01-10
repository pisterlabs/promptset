import openai
from openai import OpenAIError
import streamlit as st
import streamlit_analytics
import requests

#resume data
max_file = "https://raw.githubusercontent.com/maxwellknowles/portfolio_project/main/max.txt"
max_data = requests.get(max_file).text

#getting keys
openai.api_key = st.secrets["openai_key"]

#functions
def max(prompt):
    with st.spinner('Generating response...'):
        try:
            conversation = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": "I am a chat bot that can speak to the qualities and abilities of Maxwell Knowles, a product manager and thinker who has quickly gained experience across a number of business models and domains in the early to growth stage startup world in only three years. Maxwell believes in lightweight testing and interviews as a path to better UX, less technical debt, and stronger extensibility. He's fairly technical and loves developing product convications in the empirical. He likes to stitch a business model together with features, users, and mission. Here's Maxwell's resume: "+max_data},
                    {"role": "user", "content": prompt}
                ]
            )
            response = conversation['choices'][0]['message']['content']
        except OpenAIError as e:
            response = st.warning("Whoops! Looks like too many people were asking about Max...or, more likely, OpenAI had a problem. Try again!")
    return response

#page setup
st.set_page_config(page_title="Ask About Max", page_icon=":rocket:", layout="centered")

background_color = "black"
text_color = "white"

st.title("Ask About Max :eyes:")

response = ""

streamlit_analytics.start_tracking()
prompt = st.text_input("What would you like to ask?", placeholder="What are some of Max's best accomplishments?")

if st.button("Submit"):
    response = max(prompt)

if response != "":
    styled_text = f'<div style="background-color: {background_color}; padding: 40px; border-radius: 5px;"><span style="color: {text_color};">{response}</span></div>'
    st.markdown(styled_text, unsafe_allow_html=True)

streamlit_analytics.stop_tracking()

