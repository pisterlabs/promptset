import pandas as pd
import numpy as np
import langchain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def answer_faq(query,llm,docstore):

    docs = docstore.similarity_search(query)

    context = "\n".join([x.metadata['solution'] for x in docs])

    template = """[INST]

You are AirIndia customer support. Answer the customer's question from the given context.

context : {context}


question : {question}

[/INST]
"""

    prompt = PromptTemplate(template=template,input_variables=['context','question'])

    chain = LLMChain(prompt=prompt,llm=llm)

    return chain({'context':context,'question':query})