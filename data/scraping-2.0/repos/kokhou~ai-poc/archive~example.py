from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from langchain.callbacks import get_openai_callback

import urllib.parse
from dotenv import load_dotenv
import os
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

load_dotenv()

db = SQLDatabase.from_uri("mysql+pymysql://root:" + urllib.parse.quote('P@ssw0rd') + "@localhost:3306/merchant")
# llm = OpenAI(temperature=0, verbose=True)
# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))
agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
"""
What shop you want to open
Location
          I want you act as a business advisor.
          You will provide take the average monthly rental, average renovation cost base on different business
          Output format:
            Business Type: 
            Average Monthly Rental:
            Average Renovation Cost:
            Average Daily Foot fall:
            Average Peak Hour Foot Fall:
"""
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
            I want you act as a Business Advisor
         """
         ),
        ("user", "{question}\n ai: "),
    ]
)
# print(agent_executor.run("What is the best business is better to open in arkadia?"))
"""
Given an input question, first create a syntactically correct mysql query to run, then look at the results of the query and return the answer.
Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: 
    Business Type: 
    Average Monthly Rental:
    Average Renovation Cost:
    Average Daily Foot fall:
    Average Peak Hour Foot Fall:

I want open Bakery and location in Arkadia
"""
print(final_prompt.format(
    question="I want to know the average monthly rental fee in Arkadia."
))
agent_executor.run(final_prompt.format(
    question="I want to know the average monthly rental fee in Arkadia."
))
# agent_executor.run(
# """
# Given an input question, first create a syntactically correct mysql query to run, then look at the results of the query and return the answer.
# Use the following format:
#
# Question: Question here
# SQLQuery: SQL Query to run
# SQLResult: Result of the SQLQuery
# Answer:
#
# I want to know the average monthly rental fee in Arkadia.
# """)
# print(agent_executor.run(final_prompt.format(
#         question="I want open Bakery and location in Arkadia"
#   )))
# if not relevant to database column,
#           pick 3 following topics but not repeat: Market Research, Business Plan,
#           Location, Legal and Regulatory Requirements, Financial Planning, Marketing and Branding,
#           Inventory and Suppliers, Staffing, Operations, Customer Experience, Risk Management, Sustainability,
#           Adaptability, Competitive Analysis, Pricing Strategy, Technology and Automation, Security Measures,
#           Health and Safety Compliance, Marketing Budget Allocation, Social Media Strategy,
#           Customer Feedback and Surveys, Employee Training and Development, Exit Strategy Planning,
#           to advise me step by step. and nothing else.