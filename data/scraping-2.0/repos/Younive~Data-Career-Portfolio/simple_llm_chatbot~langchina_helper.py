import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(openai_api_key=openai_api_key, temperature=0.1) #don't be very creative pls


embeddings = OpenAIEmbeddings() #using OpenAI embeding model
vectordb_file_path = "faiss_index"

def create_vector_db():
    loading = CSVLoader(file_path='codebasics_faqs.csv',source_column='prompt') #loading data
    data = loading.load()
    vectordb = FAISS.from_documents(documents=data,embedding=embeddings) #using FAISS as vector database
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        chain_type_kwargs = {"prompt": PROMPT}
        )

    return chain

# testing
# if __name__ == "__main__":
#     chain = get_qa_chain()

#     print(chain("do you provide internship?"))


