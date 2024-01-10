import os
import urllib.parse

import openai
from dotenv import load_dotenv
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.callbacks import get_openai_callback
from langchain.llms.openai import OpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.sql_database import SQLDatabase

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY"),

hostname = 'localhost'
port = 3306  # default MySQL port
username = 'root'
password = urllib.parse.quote('P@ssw0rd')
database_name = 'poc'


def go_database():
    db = SQLDatabase.from_uri(f'mysql+pymysql://{username}:{password}@{hostname}:{port}/{database_name}')
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0, verbose=True)
    final_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("""
            You are an assistant to help me find nearest clinic by my locations and to match additional information provided by me.
            You can only answer my question using the database data I provided, if you don't find the answer just answer 'I do not know your question'
            Use clinics and services tables only.
            
            lat=lat2−lat1 (difference in latitude)
            long=long2−long1 (difference in longitude)
            R is the radius of the Earth (mean radius = 6,371 km)
            lat1 and long1 are the coordinates of the first point
            lat2 and long2 are the coordinates of the second point
            The result distance must be in kilometers.
            The closest to the user is the answer.
            
            You can only Observation these tables: clinics, services
            SELECT limit 3 only
            """),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    ).format(
        question="""
            I'm looking for Emergency Dental Care Service
            My Location Latitude: 3.04885
            My Location Longitude: 101.5592222
        """
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    with get_openai_callback() as cb:
        response = agent_executor.run(final_prompt)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    print(response)


go_database()
