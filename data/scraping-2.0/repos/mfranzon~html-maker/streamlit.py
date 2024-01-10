import openai
import streamlit.components.v1 as components 
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="HTML maker",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="auto",
)

def openai_completion(prompt, token, generated_tokens):
    openai.api_key = token
    response = openai.Completion.create(
      model="code-davinci-002",
      prompt=prompt,
      max_tokens=generated_tokens,
      temperature=0,
      n=1
    )
    return response['choices'][0]["text"]


with st.sidebar:
    token = st.text_input("Open AI Token", placeholder="xxxxxxxxx")
    generated_tokens = st.slider("Max number of response tokens", 100, 500)

st.title("HTML maker")

input_text = st.text_area("Please enter text here... 🙋",height=50)
chat_button = st.button("Do the Magic! ✨")
if chat_button and input_text.strip() != "":
    with st.spinner("Loading...💫"):
        try:
            openai_answer = openai_completion(input_text + "<!DOCTYPE html>", token=token, generated_tokens=generated_tokens)
            html_soup = BeautifulSoup(openai_answer.split("</html>")[0] + "</html>", "html.parser")
            col1, col2 = st.columns(2)
            with col1:
                st.code(html_soup.prettify(), language="html")
            with col2:
                components.html(openai_answer.split("</html>")[0] + "</html>", width=300, height=300)

        except Exception as e:
            st.warning("Please insert a valide OpenAI token" + str(e))
      
else:
    st.subheader("Some ideas for a quick try:")
    st.code("Create a warning button", language="plain-text")
    st.code("Create a dropdown menu", language="plain-text")
    st.code("Create a login form", language="plain-text")






