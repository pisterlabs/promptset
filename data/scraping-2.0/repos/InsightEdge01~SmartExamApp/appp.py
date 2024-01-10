import streamlit as st
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Replicate
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

prompt_template_questions = """
You are an expert in creating practice questions based on study material.
Your goal is to prepare a student for their exam. You do this by asking questions about the text below:

------------
{text}
------------

Create questions that will prepare the student for their exam. Make sure not to lose any important information.

QUESTIONS:
"""

PROMPT_QUESTIONS = PromptTemplate(template=prompt_template_questions, input_variables=["text"])

refine_template_questions = """
You are an expert in creating practice questions based on study material.
Your goal is to help a student prepare for an exam.
We have received some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones.
(only if necessary) with some more context below.
------------
{text}
------------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.

QUESTIONS:
"""

REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_questions,
)

# Initialize Streamlit app
st.title('Question Answering Generator:books:')
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
st.subheader('Built for Professionals,Teachers, Students')
st.markdown('<style>h3{color: pink;  text-align: center;}</style>', unsafe_allow_html=True)

# File upload widget
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

# Set file path
file_path = None

# Check if a file is uploaded
if uploaded_file:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name

# Check if file_path is set
if file_path:
    # Load data from the uploaded PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # Combine text from Document into one string for question generation
    text_question_gen = ''
    for page in data:
        text_question_gen += page.page_content
    
     # Initialize Text Splitter for question generation
    text_splitter_question_gen = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=50)

    # Split text into chunks for question generation
    text_chunks_question_gen = text_splitter_question_gen.split_text(text_question_gen)

    # Convert chunks into Documents for question generation
    docs_question_gen = [Document(page_content=t) for t in text_chunks_question_gen]
    
    # Initialize Large Language Model for question generation
    llm_question_gen = Replicate(
        model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
        input={"temperature": 0.01, "max_length": 500, "top_p": 1}
    )

    # Initialize question generation chain
    question_gen_chain = load_summarize_chain(llm=llm_question_gen, chain_type="refine", verbose=True,
                                              question_prompt=PROMPT_QUESTIONS, refine_prompt=REFINE_PROMPT_QUESTIONS)
    # Run question generation chain
    questions = question_gen_chain.run(docs_question_gen)

    # Initialize Large Language Model for answer generation
    llm_answer_gen = Replicate(
        model="replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781",
        input={"temperature": 0.01, "max_length": 500, "top_p": 1})
    
    # Create vector database for answer generation
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

    # Initialize vector store for answer generation
    vector_store = Chroma.from_documents(docs_question_gen, embeddings)

    # Initialize retrieval chain for answer generation
    answer_gen_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, chain_type="stuff",
                                                   retriever=vector_store.as_retriever(k=2))
    
     # Split generated questions into a list of questions
    question_list = questions.split("\n")

    # Answer each question and save to a file
    for question in question_list:
        st.write("Question: ", question)
        answer = answer_gen_chain.run(question)
        st.write("Answer: ", answer)
        st.write("--------------------------------------------------\n\n")

    # Create a directory for storing answers
    answers_dir = os.path.join(tempfile.gettempdir(), "answers")
    os.makedirs(answers_dir, exist_ok=True)

    # Create a single file to save questions and answers
    qa_file_path = os.path.join(answers_dir, "questions_and_answers.txt")

    with open(qa_file_path, "w") as qa_file:
        # Answer each question and save to the file
        for idx, question in enumerate(question_list):
            answer = answer_gen_chain.run(question)
            qa_file.write(f"Question {idx + 1}: {question}\n")
            qa_file.write(f"Answer {idx + 1}: {answer}\n")
            qa_file.write("--------------------------------------------------\n\n")

    # Create a download button for the questions and answers file
    st.markdown('### Download Questions and Answers')
    st.download_button("Download Questions and Answers", qa_file_path)

# Cleanup temporary files
if file_path:
    os.remove(file_path)