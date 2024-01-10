from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from text_chunker import get_text_chunks


from langchain.document_loaders import DirectoryLoader
load_dotenv()

llm = GooglePalm(google_api_key=os.getenv("GOOGLE_API_KEY"), temperature = 0)

embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path_SRT = "knowledge_base/SRT"
vectordb_file_path_TXT = "knowledge_base/TXT"

def create_vector_db():
    loader_SRT = CSVLoader(file_path="knowledge_base/CSV/output.csv", source_column="Timecode")
    docs_SRT = loader_SRT.load()
        
    text_loader_kwargs={'autodetect_encoding': True}
    loader_TXT = DirectoryLoader(vectordb_file_path_TXT, glob="**/*.txt", loader_kwargs=text_loader_kwargs, show_progress=True, use_multithreading=True)
    docs_TXT = loader_TXT.load()
    chunks = get_text_chunks(docs_TXT)
    
    vectordb = FAISS.from_documents(documents = docs_SRT, embedding= embeddings)
    vectordb.save_local(vectordb_file_path_SRT)
    
    vectordb = FAISS.from_documents(documents = chunks, embedding= embeddings)
    vectordb.save_local(vectordb_file_path_TXT)
    
def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path_TXT, embeddings)
    
    retriever = vectordb.as_retriever()
    
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain
    

if __name__ == "__main__":
    if not os.path.exists(vectordb_file_path_TXT) or not os.listdir(vectordb_file_path_TXT):
        print("Hi")
        create_vector_db()
    # chain = get_qa_chain()
    # print(chain("What is Parido principle?"))