from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os  # Added for file existence check

vectordb_file_path = "faiss_index"
api_key = "enter your api key"
llm = GooglePalm(google_api_key=api_key, temperature=0.001)
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")


def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path=r'llm_dataset.csv', source_column="prompt")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain(vectordb):
    # Use the passed vectordb, no need to reload
    retriever = vectordb.as_retriever(score_threshold=0.7)

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
                                        return_source_documents=False,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain


def ask_question(chain_instance, query):
    # user_query = input("Please enter your question: ")
    response = chain_instance(query)
    return response['result']


if __name__ == "__main__":
    if not os.path.exists(vectordb_file_path):  # Only create vector db if it doesn't exist
        create_vector_db()

    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)  # Load the vector database once
    chain_instance = get_qa_chain(vectordb)  # Create the chain instance once using the loaded vector database

    print(ask_question(chain_instance, "what's edema"))  # Ask the question
