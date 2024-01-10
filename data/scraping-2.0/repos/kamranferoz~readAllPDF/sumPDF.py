import os
import streamlit as st
import openai
import PyPDF2
import config

# Set your OpenAI API key here
openai.api_key = config.OPENAI_API_KEY

def generate_summary(text):
    response = openai.Completion.create(
        engine="davinci",
        prompt=text,
        max_tokens=100,  # Adjust the length of the summary as needed
        temperature=0.7  # Adjust the creativity of the summary
    )
    return response.choices[0].text.strip()

def main():
    st.title("PDF File Summary App")
    
    # Select a folder containing PDF files
    folder_path = st.sidebar.text_input("Enter folder path:", key="folder_path")
    
    if folder_path and os.path.isdir(folder_path):
        st.write(f"Selected folder: {folder_path}")
        files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]
        
        for file in files:
            st.write(f"### {file}")
            file_path = os.path.join(folder_path, file)
            try:
                pdf_reader = PyPDF2.PdfReader(file_path)
                file_contents = ""
                for page_num in range(len(pdf_reader.pages)):
                    file_contents += pdf_reader.pages[page_num].extract_text()
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
                continue
            
            st.write("File contents:")
            st.write(file_contents)  # Display the file contents for debugging
            
            try:
                summary = generate_summary(file_contents)
                st.write("Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"Error generating summary: {e}")

if __name__ == "__main__":
    main()

