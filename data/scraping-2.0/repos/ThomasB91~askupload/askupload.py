import streamlit as st
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Initialize SentenceTransformer
model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

# Streamlit interface
st.title("File Vectorization and Question Answering App")

# File upload
uploaded_file = st.file_uploader("Upload a file for vectorization")
question = st.text_input("Ask a question about the file")

# Process file and answer questions
if uploaded_file is not None and question:
    # Read file content
    file_content = uploaded_file.getvalue().decode()

    # Vectorize content
    vectorized_content = model.encode([file_content])

    # Use OpenAI API securely
    OPENAI_API_KEY = st.secrets["openai_api_key"]
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    )

    # Extracting and displaying the answer
    answer = response['choices'][0]['message']['content']
    st.write("Answer:", answer)

# Run this with `streamlit run your_script.py`