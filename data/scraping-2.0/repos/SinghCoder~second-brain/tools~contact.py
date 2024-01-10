import json
from datetime import datetime
import parsedatetime as pdt
from langchain.tools import tool
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

@tool
def get_contact(query: str) -> str:
    '''
    Given a query string, return the user's information based on that query.
    Example: "Harpinder's phonenumber?"
    
    Returns:
    A string containing the user's information.
    '''
    try:
        f = open('store/users.md', 'r')
        users = f.read()
        f.close()
        llm = OpenAI(temperature=0)
        input = ["users", "question"]
        template = "{users}\nBased on the above data, answer the following: {question}?"
        prompt = PromptTemplate(input_variables=input, template=template)
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run({'users': users, 'question': query})
    except Exception as e:
        return json.dumps({'error': str(e)})

