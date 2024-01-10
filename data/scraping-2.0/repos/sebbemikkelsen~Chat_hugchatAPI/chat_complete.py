from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
import os

load_dotenv()


def get_vectorstore(embedding_model, db_name):
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
    db = Chroma(persist_directory=db_name, embedding_function=embeddings)
    return db



custom_prompt_template = """Use the context to answer the user's question. Use the history to know what it's about.
If you don't know the answer, say that you don't know, don't try to make up an answer.

Chat history: {history}
Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['history', 'context', 'question'])
    return prompt


#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={
                                           "verbose": True,
                                           'prompt': prompt, 
                                           "memory": ConversationBufferMemory(memory_key="history", input_key="question")
                                        }
                                       )
    return qa_chain


#Loading the model
def load_llm():
    # Load the locally downloaded model here
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

    repo_id = "google/flan-t5-xxl"
    #repo_id = "google/mt5-base"

    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "min_tokens": 200})

    return llm




#QA Model Function
def qa_bot():
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    db = get_vectorstore(embedding_model, "chroma_db")
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa





#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response





def main():

    #query = "How many books are there in the library?"
    #res = final_result(query)

    #print(res)


    qa_result = qa_bot()
    while True:
        query = input("Question: ")
        if query == "quit":
            break

        response = qa_result({'query': query})

        print(response)








if __name__=='__main__':
    main()

