from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

from dotenv import load_dotenv
load_dotenv()

# Create Google Palm LLM model
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1 )

# Initialize instructor embeddings using the Hugging Face model and mention vectorbd_path
instruct_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index" # it will create folder directory

def create_vector():
    # load the CSV file
    loader = CSVLoader(file_path='CSV_FILE', source_column = "prompt")
    data = loader.load()

    # Create embedding by sung FAISS_db
    vectordb = FAISS.from_documents(documents=data, embedding=instruct_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)

def QA_retriever ():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instruct_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever()

    template = """Given the following context and question, generate the answer based on the context only.
    try to provide as much text as possible from "response" section in the source document context without make things up.
    is not found in the context, kindly state "I dont know." Don't try to make up an answer.
    
    CONTEXT : {context},
    QUESTION : {question}
    
    """

    prompt = PromptTemplate(template=template, input_variables= ["context", "question"])

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                    retriever = retriever,
                                    input_key = "query",
                                    return_source_documents = True,
                                    chain_type_kwargs = {"prompt":prompt})

    return chain
print("hello")

if __name__ == "__main__":
    # create_vector()
    question = QA_retriever()

    print(question("Do you have javascript course?"))
