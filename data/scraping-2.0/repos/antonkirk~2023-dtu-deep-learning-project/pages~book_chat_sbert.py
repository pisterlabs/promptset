from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Chroma
from pypdf import PdfWriter
import streamlit as st

def parse_txt(file, file_name):
    doc = file.read().decode()
    file_path = f'documents/{file_name}'

    with open(file_path, 'w') as f:
        f.write(doc)

    loader = TextLoader(file_path)
    return loader

def parse_pdf(file, file_name):
    pdf_writer = PdfWriter()

    pdf_writer.append(file)

    file_path = f'documents/{file_name}'

    with open(file_path, 'wb') as out:
        pdf_writer.write(out)

    loader = PyPDFLoader(file_path)

    return loader

def embed_document(loader):
    text_splitter = RecursiveCharacterTextSplitter(separators= ["\n\n", "\n", "\t"], chunk_size=1500, chunk_overlap=500)
    splits = text_splitter.split_documents(loader.load())

    model_name = "dtu-deep-learning-course-f2023/msmarco-rag-finetune"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    return vectorstore

st.set_page_config(
        page_title="BookChat",
)

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

st.title("📖 BookChat with fine-tuned embeddings")
st.caption("Ask your textbook questions using GPT-3.5 Turbo.")


uploaded_file = st.file_uploader("Upload a chapter from your textbook, or a course description", type=("txt","pdf"))

question = st.text_input(
    "Ask something about the document",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question and not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")

if uploaded_file and question and openai_api_key:
    document_name = uploaded_file.name
    document_type = uploaded_file.type
    if document_type == 'text/plain':
        text = parse_txt(uploaded_file, document_name)
    elif document_type == 'application/pdf':
        text = parse_pdf(uploaded_file, document_name)
    else:
        st.error('Unsupported file type')

    vectorstore = embed_document(text)
    retriever = vectorstore.as_retriever()

    TEMPLATE = """ \
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise, but thorough.
    Question: {question}
    Context: {context}
    Answer:
    """

    rag_prompt = PromptTemplate.from_template(template=TEMPLATE)

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1, openai_api_key=openai_api_key)

    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser()

    docs = vectorstore.similarity_search(question)

    for doc in docs:
        print(doc)

    st.write(rag_chain.invoke(question))
    
    st.write("Sources:")
    for i, doc in enumerate(docs):
        source = doc.metadata["source"]
        st.write(f"{i}: {source}")
