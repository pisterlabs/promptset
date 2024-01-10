import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import openai
import tempfile


# Initialize Streamlit
st.title("PEARL Strategy for Article Summarization")

# Load environment variables
load_dotenv()

# Load Open AI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# File Upload
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])



# File processing and PEARL Strategy
# File processing and PEARL Strategy
if uploaded_file is not None:
    st.write("File successfully uploaded. Press the button to start the PEARL strategy.")
    if st.button("Start PEARL"):
    # Show different loaders for the different sections with text showing what is being created, for exaple "Loading the PDF document" and "Now creating What is the article about?"
        st.write("Loading the PDF document...")
        st.write("Now creating What is the article about?")
        st.write("Now creating What methodologies are used?")
        st.write("Now creating What are the key findings?")
        st.write("Now creating What recommendations are made?")
        st.write("Now creating the PEARL summary...")
        
        
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            fp.write(uploaded_file.getvalue())
            temp_file_path = fp.name

        # Load the PDF document using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        data = loader.load()
        
        # Split the text into chunks using TokenTextSplitter
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(data)

        # Generate embeddings using OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()

        # Create a retriever using Chroma and the generated embeddings
        retriever = Chroma.from_documents(chunks, embeddings).as_retriever()

        # Initialize the ChatOpenAI model
        llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.1)

        # Create a RetrievalQA instance with the ChatOpenAI model and the retriever
        qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

        # Predefined questions for PEARL analysis
        predefined_questions = [
            "What is the article about?",
            "What methodologies are used?",
            "What are the key findings?",
            "What recommendations are made?"
        ]

        # Define the PEARL strategy function
        def pearl_strategy(question, context, llm):
            # PEARL Decomposition: Break down the question into sub-questions
            sub_questions = [
                f"Step 1: What is the main theme of the article regarding {question}?",
                f"Step 2: What are the key points in the article related to {question}?",
                f"Step 3: What actions does the article suggest about {question}?"
            ]
            
            # Execute the steps and collect the answers
            answers = []
            for sub_q in sub_questions:
                prompt = f"{sub_q}\n{context}"
                
                # Replace the following line with the correct method for generating text
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",  # replace with the model you are using
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.choices[0].message.content.strip()
                
                answers.append(answer)

            # Compile the answers into a coherent summary
            summary = " ".join(answers)
            return f"Question: {question}\nPEARL Summary: {summary}"


        # Automated PEARL Analysis
        for question in predefined_questions:
            # Use the retriever to get relevant chunks
            retrieved_docs = retriever.get_relevant_documents(question)
            
            # Concatenate the retrieved chunks to form the context
            context = " ".join([doc.page_content for doc in retrieved_docs])
            
            # Use the PEARL strategy to answer the question
            pearl_summary = pearl_strategy(question, context, llm)
            
            st.write(f"### {question}")
            st.write(pearl_summary)
            st.write("------\n")