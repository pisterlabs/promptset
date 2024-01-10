# from langchain import PromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from third_parties.linkedin import scrape_linkedin_profile
from dotenv import load_dotenv
import os
import data.accounts

template = """
read all my accounts {accounts} and produce the following:
1.list my uber expenses in order of date  
2.add them all up
3.calculate the average cost
"""

load_dotenv()
if __name__ == "__main__":
    template_info = PromptTemplate(input_variables=["accounts"], template=template)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
chain = LLMChain(llm=llm, prompt=template_info)
print(chain.run(accounts=data.accounts.accounts))
