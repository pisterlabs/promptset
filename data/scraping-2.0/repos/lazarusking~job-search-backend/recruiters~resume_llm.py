# !pip install langchain
# !pip install unstructured
# !pip install openai
# !pip install chromadb
# !pip install Cython
# !pip install tiktoken
# !pip install pdf2image
# !pip install pdfminer.six

import os
from langchain.document_loaders import PDFMinerLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


def create_index(pdf_folder_path: list[str]):
    # Load the PDF files from the specified folder
    print([i.path for i in pdf_folder_path])
    def add_end(file):
        loader=PDFMinerLoader(os.path.join(file.path))
        for i in loader.load():
            # print(i.page_content)
            i.page_content="".join(f"{i.page_content},<End>")
        return loader
    loaders = [add_end(file)
               for file in pdf_folder_path]

    # Create and return a VectorstoreIndex from the PDF loaders
    index = VectorstoreIndexCreator(text_splitter=RecursiveCharacterTextSplitter(separators='<End>',
        chunk_size=len(loaders)*1000, chunk_overlap=0)).from_loaders(loaders)
    print(loaders)
    return index


def prompt_decorator(func):
    def wrapper(*args, **kwargs):
        job_desc = func(*args, **kwargs)
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        You are a chatbot that helps recruiters analyze resumes based on a job description. You will be passed information on some resumes as the context, try your best to answer accurately about them based on the job description also note the full name of the person in each resume.

        Keep the answer as concise as possible. 
        Always say "thanks for asking!" at the end of the answer. 
        {context}

        Job Description: {job_desc}
        Question: {question}
        Answer:"""
        return prompt_template.format(job_desc=job_desc, context='{context}', question='{question}')
    return wrapper


def load_qa_chain_with_prompt(llm, job_desc):
    # Define the prompt template for the QA chain
    @prompt_decorator
    def my_func(desc):
        return desc

    result = my_func(job_desc)
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    You will be passed documents about resume, try your best to answer accurately about them based on this job description.
    Use three sentences maximum and keep the answer as concise as possible. 
    Always say "thanks for asking!" at the end of the answer. 

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(template=result,
                            input_variables=["context", "question"])

    return load_qa_chain(llm=llm, chain_type="stuff", prompt=PROMPT, verbose=True)


def search_resumes(index, query: str, job_desc: str):
    # Set the OpenAI API key
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo",
                     streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

    # Create the VectorstoreIndex

    # Load the QA chain with the specified prompt template
    qa_chain = load_qa_chain_with_prompt(llm, job_desc)

    # Create a RetrievalQA instance with the QA chain and index retriever
    qa = RetrievalQA(combine_documents_chain=qa_chain,
                     retriever=index.vectorstore.as_retriever())

    # Run the query and return the result
    result = qa.run(query)
    # return qa({"query": query,"job_desc":job_desc})['result']
    return result
