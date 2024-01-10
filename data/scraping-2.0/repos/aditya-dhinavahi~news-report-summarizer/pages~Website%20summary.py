import streamlit as st
import time
from datetime import datetime
import pandas as pd

from dotenv import load_dotenv

from langchain.callbacks import get_openai_callback

from llm_functions import get_text_from_url, get_text_chunks, get_text_summary, get_text_summary_custom

load_dotenv()

st.set_page_config(page_title = "Website summarizer",
                    page_icon = ":newspaper:", layout = "wide")

st.header("Summarize any website content")
st.text('''This app helps you summarize any website text. Enter URL, and choose which type of summary you want 
        ''')

# prompt_user_text = ""
# chunk_size = None
# chunk_overlap = None


url = st.text_input('''#### Enter single url to get article summary for:''', 
                                value = 'https://collabfund.com/blog/intelligent-vs-smart/')
# def pp():

#     global prompt_user_text
#     global chunk_size
#     global chunk_overlap

    
            

summary_type = st.radio('''#### Select the type of summary you want''',
                        ["simple", "topic-wise"])
                        # captions = ["for one-page contentt; 160 words output", 
                        #             "for multi-page content; 160 words output per topic"])

if summary_type == "simple":
    prompt_user_text = st.text_area('''#### Enter the prompt text''', 
                                value = "You are a smart editorial assistant who has to Write a summary the article below. \
The summary is meant to be read by analysts who are smart and intelligent people. \
So write in a clear and concise manner, don't use verbose language. \
No need to state the obvious by starting the sentence with 'This text...', just get right to the point. \
\
Other key rules to follow -\
Keep the length of the summary to max 160 words.", height = 100)
    chunk_size = 2000,
    chunk_overlap = 500
elif summary_type == "topic-wise":
    prompt_user_text = st.text_area('''#### Enter the prompt text''', 
                                    value = "You are a smart editorial assistant who has to Write a summary the article below. \
The summary is meant to be read by analysts who are smart and intelligent people. \
So write in a clear and concise manner, don't use verbose language. \
No need to state the obvious by starting the sentence with 'This text...' or 'This article...', just start with verbs like 'highlights' or 'says' or 'describes' etx. \
just get right to the point.Write topic-wise summaries for the text below. \
\
Other key rules to follow  - \
- Get max 10 key topics from the text and write a summary for each topic.\
- Each topic summary should be max 160 words.\
                                ", height = 150)
    chunk_size = 500
    chunk_overlap = 125
# elif(summary_type == "custom"):
#      prompt_user_text = form.text_area('''#### Enter the prompt text''', 
#                                  value = "")

submit2 = st.button('''Get summary''')

if submit2:
    output_text = ""
    try:
        text = get_text_from_url(url)
        text_chunks = get_text_chunks(text, chunk_size, chunk_overlap)
        text_summary = get_text_summary_custom(summary_type, prompt_user_text, text_chunks)
        output_text = output_text + str(text_summary) + "\n"
        st.write(str(text_summary))
        output_text = output_text + "\n \n"
    except Exception as e:
        st.write("Error in summarizing article")
        st.write(e)
