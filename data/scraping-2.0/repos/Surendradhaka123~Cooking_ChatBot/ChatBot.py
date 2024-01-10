import streamlit as st
import time
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
import sys
import os
from datetime import datetime
# from streamlit_chat import message as st_message


html_temp= """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:centre;">(●'◡'●) Cook with me (●'◡'●) </h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

st.markdown(
   
    """
    This is an AI chatBot it is based on Cooking interview. You can ask it some general questions and cooking related questions.
"""
)
# Code 
os.environ["OPENAI_API_KEY"] = st.secrets["KEY"]
index = GPTSimpleVectorIndex.load_from_disk('index.json')



Warning="By selecting the check box you are agree to use our app.\nDon't worry!! We will not save your any data."
check=st.checkbox("I agree",help=Warning)
if(check):
    st.write('Great!')
    btn=st.button("Start")
    st.write('Enter "quit" for end the chat')
    if btn:
        def ask_question():
            count=0
            
            while True:
                query = st.text_input("Type your question..",key=count)
                count= count+1

                if(query=="quit"):
                    break
                else:
                    response = index.query(query)
                    st.markdown(f"Response: \t{response.response}\t")
                    
        ask_question()
        st.text("Thanks for using")
            
if st.button("About"):
        st.text("Created by Surendra Kumar")
## footer
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):
    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="solid",
        border_width=px(0.5)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )
    st.markdown(style,unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "©️ surendraKumar",
        br(),
        link("https://www.linkedin.com/in/surendra-kumar-51802022b", image('https://icons.getbootstrap.com/assets/icons/linkedin.svg') ),
        br(),
        link("https://www.instagram.com/im_surendra_dhaka/",image('https://icons.getbootstrap.com/assets/icons/instagram.svg')),
    ]
    layout(*myargs)

if __name__ == "__main__":
    footer()
