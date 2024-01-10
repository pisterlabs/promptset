from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def user_query(knowledge_base, user_question):
    load_dotenv()
    
    if not user_question is None and user_question != "":
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            # price is save in postgresql database
            