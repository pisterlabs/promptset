import pdfplumber
import streamlit as st
import openai
import io
openai.api_key = 'KEY'
model_name = 'gpt-3.5-turbo'
def search_openai(query,num=5):
    response = openai.Completion.create(
        engine='text-davinci-003',  # Use the GPT-3.5 engine
        prompt=query,
        max_tokens=100,  
        temperature=0.7, 
        n=num,
        stop=None,
        timeout=10, 
    )
    return response.choices[0].text.strip()
def summarize_pdf(file_path, num_sentences=3):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    summary = search_openai("summerize this text"+text)
    return summary

def pdfsummerizermain():
    st.title("Idea-File-Compress")
    # File uploader widget
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        # Convert the bytes object to a file-like object
        file_obj = io.BytesIO(uploaded_file.read())
        # Process the PDF contents and generate summary
        summary = summarize_pdf(file_obj)
        st.write(summary)
if __name__=='__main__':
    pdfsummerizermain()
