from prompt import PROMPT
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def create_chain(llm):
    qa_chain_prompt = PromptTemplate(input_variables=["question"],template=PROMPT,)    
    llm_chain = LLMChain(
        llm=llm,
        prompt=qa_chain_prompt
    )
    return llm_chain




