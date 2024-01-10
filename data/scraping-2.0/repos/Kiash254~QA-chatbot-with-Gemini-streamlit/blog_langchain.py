import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain.schema import HumanMessage

# Load the environment variables from a .env file
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# Function to get response from Gemini model
def getGeminiResponse(input_text, no_words, blog_style):
    # Prompt Template
    template="""
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
            """
    
    prompt = template.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
    
    # Generate the response from the Gemini model
    response = llm([HumanMessage(content=prompt)])
    return response.content

# Streamlit UI
st.set_page_config(page_title="Generate Blogs", page_icon='🤖', layout='centered', initial_sidebar_state='collapsed')
st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

# Creating two more columns for additional 2 fields
col1, col2 = st.columns([5,5])

with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)
    
submit = st.button("Generate")

# Final response
if submit:
    st.write(getGeminiResponse(input_text, no_words, blog_style))