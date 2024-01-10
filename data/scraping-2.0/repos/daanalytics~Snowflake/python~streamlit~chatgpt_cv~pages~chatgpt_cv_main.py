# Importing required packages
import streamlit as st
import openai
import os

# NLP Packages from textblob import TextBlob
import spacy from gensim. summarization. summarizer import summarize import nltk
nitk.download('punkt' )

# Sumy Summary Packages from sumy. parsers. plaintext import PlaintextParser from sumy.nlp.tokenizers import Tokenizer
from sumy. summarizers. lex_ rank import LexRankSummarizer

st.title("Chatting with ChatGPT")
st.sidebar.header("Instructions")
st.sidebar.info(
    '''This is a web application that allows you to interact with 
       the OpenAI API's implementation of the ChatGPT model.
       Enter a **query** in the **text box** and **press enter** to receive 
       a **response** from the ChatGPT
       App is based on the following blogs:
       https://blog.devgenius.io/building-a-chatgpt-web-app-with-streamlit-and-openai-a-step-by-step-tutorial-1cd57a57290b
       '''
    )

# Set the model engine and your OpenAI API key
model_engine = "text-davinci-003"
openai.api_key = "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" #follow step 4 to get a secret_key

def main():
    '''
    This function gets the user input, pass it to ChatGPT function and 
    displays the response
    '''
    # Get user input
    user_query = st.text_input("Enter query here, to exit enter :q", "what is Python?")
    if user_query != ":q" or user_query != "":
        # Pass the query to the ChatGPT function
        response = ChatGPT(user_query)
        return st.write(f"{user_query} {response}")
    
def ChatGPT(user_query):
    ''' 
    This function uses the OpenAI API to generate a response to the given 
    user_query using the ChatGPT model
    '''
    # Use the OpenAI API to generate a response
    completion = openai.Completion.create(
                                  engine = model_engine,
                                  prompt = user_query,
                                  max_tokens = 1024,
                                  n = 1,
                                  temperature = 0.5,
                                      )
    response = completion.choices[0].text
    return response

# call the main function
main() 
