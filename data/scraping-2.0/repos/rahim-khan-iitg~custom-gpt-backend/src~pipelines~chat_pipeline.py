from utils.utils import load_LLM,connect_to_qdrant_as_retriver,connect_to_chroma_as_retriver
from langchain.chains import RetrievalQA,LLMChain
from langchain.prompts import PromptTemplate

class ChatModel:
    """chat without database"""
    def __init__(self) -> None:
        pass

    def chat(self,question:str):
        llm=load_LLM()
        template = """Question: {question} Answer:"""
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response=llm_chain.run(question)
        # print("res= ",response)
        return response

class QueryModel:
    """chat with database"""
    def __init__(self) -> None:
        pass

    def chat(self,question:str):
        llm=load_LLM()
        retriever1=connect_to_qdrant_as_retriver()
        # retriever1=connect_to_chroma_as_retriver()
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever1, chain_type="stuff",return_source_documents=False)
        response=qa(question)
        return response