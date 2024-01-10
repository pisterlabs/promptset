import sys
import os
sys.path.append(f'{os.path.dirname(__file__)}/../..')

from botcore.setup import trace_ai21
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate



TEMPLATE = """Given the question: {question}.
What is the product mentioned in the given question?
Answer the product name:"""

def build_extract_product_chain(model):
    
    prompt = PromptTemplate(input_variables=["question"], template=TEMPLATE)
    chain = LLMChain(llm=model, prompt=prompt)
    return chain
