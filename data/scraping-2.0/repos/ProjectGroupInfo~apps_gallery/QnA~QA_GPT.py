
import openai
import streamlit as st
from PyPDF2 import PdfReader

def load_pdf(file):
    pdf_reader = PdfReader(file)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page in range(num_pages):
        page_obj = pdf_reader.pages[page]
        text += page_obj.extract_text()
    return text

def main():    
    # Set API key as a secret
    with open('api_key.txt', 'r') as f:
        api_key = f.read().strip()
    openai.api_key = api_key    
    
    # Prompt user to upload a PDF file
    file = st.file_uploader("Upload a PDF file", type="pdf")

    # Generate summary and example QnAs if file is uploaded
    if file is not None:
        # Read PDF content
        with st.spinner('Extracting text from PDF...'):
            text = load_pdf(file)

        # Assign PDF content to prompt var
        prompt = text

        # Set up Streamlit app 
        st.title("Ask a Question about Pdf Document")

        # Prompt user to enter a question
        question = st.text_input("What do you want to know about it?")

        # Generate answer if user inputs a question
        if question:
            prompt = f"Answer the question as truthfully as possible using the provided text. If the answer is not contained within the text below, say 'I don't know'.\n\n{prompt}"
            prompt_with_question= f"{prompt}\n\nQuestion: {question}\nA"
            response = openai.Completion.create(
                prompt=prompt_with_question,
                temperature=0.0,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                model="text-davinci-003",
                stop=["Q:", "\n"],
                timeout=600  # wait for 600 seconds before timing out
            )

            # Extract answer from OpenAI API response
            answer = response.choices[0].text.strip()

            # Output answer or "I don't know" if answer is empty
            answer_output = answer.strip() if answer.strip() != '' else "I don't know"
            st.write(f"Q: {question}")
            st.write(f"A: {answer_output}\n")
                    
if __name__ == "__main__":
    st.set_page_config(page_title='PDF Question and Answer', page_icon=':books:')
    st.title('PDF Question and Answer')
    st.write('App to answer your questions about PDF document.')
    main()
