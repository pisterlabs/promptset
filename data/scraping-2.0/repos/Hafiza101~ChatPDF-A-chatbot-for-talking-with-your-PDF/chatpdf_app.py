# import libraries
import streamlit as st
import PyPDF2
import io
import openai
import docx2txt
import pyperclip

st.title("Document Reading App like ChatGPT") 


openai.api_key = ""
st.sidebar.title("Please enter your OpenAI API Key")
openai.api_key = st.sidebar.text_input("API Key", type="password")

import PyPDF2
PyPDF2.__version__ = '3.0.0'
# Defining a function to extract text from PDF file

def extract_text_from_pdf(file):
    # Creating a BytesIO object from the uploaded file 
    pdf_file_obj = io.BytesIO(file.read())
    # Creating a PDF reader object from the BytesIO object
    pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)
    #initializing an empty string to store the extracted text
    text = " "
    #Looping through each page of the PDF file
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.getPage(page_num)
        text += page.extractText()
        #Return the extracted text
    return text
# Defining a function to extract text from docx file
def extract_text_from_docx(file):
    #Creating a BytesIO object from the uploaded file
    docx_file_obj = io.BytesIO(file.read())
    #Extracting text from the Word file
    text = docx2txt.process(docx_file_obj)
    #Return the extracted text
    return text

# Defining a function to extract text from txt file
def extract_text_from_txt(file):
    #Creating a TextIOWrapper object from the uploaded file
    txt_file_obj = io.TextIOWrapper(file,encoding = 'utf-8')
    #Reading the contents of the uploaded file
    text = txt_file_obj.read()
    #Return the extracted text
    return text

# Define a function to extract text from a file basaed on its type
def extract_text_from_file(file):
    text = ""
    if file.type == 'application/pdf':
         text = extract_text_from_file(file)
    elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        #Extracting text from Word file
        text = extract_text_from_docx(file)
    elif file.type == "txt/plain":
         textext = extract_text_from_txt(file)
    else:
        st.error("This file format is not supported!")
        text = None
     # Returning the extracted text         
    return text

# Defining a function to generate questions from text using OpenAI's GPT-3
def generate_questions(text):
    #selecting the first 4096 characters of the text as the prompt for the GPT-3 API
    prompt = text[:4096]
    #Generating a question using the GPT-3 API
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, temperature=0.5, max_tokens=30)
    #Returning the generated question
    return response.choices[0].text.strip()

# Defining a function to generate answers from text using OpenAI's GPT-3
def generate_answers(question, text):
    #selecting the first 4096 characters of the text as the prompt for the GPT-3 API
    prompt = text[:4096]
    #Adding the question to the prompt
    prompt += "\nQ: " + question + "\nAnswer:"
    #Generating an answer using the GPT-3 API
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, temperature=0.6, max_tokens=2000)
    #Converting the generator object to a list and then accessing the choices attribute
    choices = list(response)[0].choices
    #Returning the generated answer
    return choices[0].text.strip()

#Defining the main function of the Streamlit app
def main():
    #Setting the title of the app
    st.title("Ask Question From Uploaded Document")
    #Creating a file uploader for PDF, Word, and Text files
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
    #Checking if the user has uploaded a file or not
    if uploaded_file is not None:
        # Extracting text from the uploaded file
        text = extract_text_from_file(uploaded_file)
        # Checking if text was extracted successfully
        if text is not None:
            #Generating a question from the extracted text using GPT-3
            question = generate_questions(text)
            #Displaying the generated question
            st.write(" Question: ", question)
            #Creating a text input for the user to ask a question
            user_question = st.text_input("Ask a question about document")
            #checking if the user has asked a question or not
            if user_question:
                #Generating an answer to the user's question using GPT-3
                answer = generate_answers(user_question, text)
                #Displaying the generated answer
                st.write("Generated Answer: ", answer)
                # Creating a button to copy the answer to the clipboard
                if st.button("Copy Answer"):
                    pyperclip.copy(answer)
                    st.success("Answer copied to clipboard!")


#Calling the main function
if __name__ == "__main__":
    main()


import sys
sys.setrecursionlimit(10000)
