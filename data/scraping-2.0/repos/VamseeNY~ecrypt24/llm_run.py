import os
from os.path import join, dirname
from dotenv import load_dotenv


from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.agents.agent_types import AgentType
from langchain_community.chat_models import ChatOpenAI

from langchain_community.llms import OpenAI
import pandas as pd


dotenv_path = join(dirname(__file__), '.env_file')
load_dotenv(dotenv_path)


OPENAI_API_KEY =os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0.8, model ='gpt-4', 
                              openai_api_key=OPENAI_API_KEY)
df = pd.read_csv("iitm2.csv")
df = df.drop("Age", axis=1)

def scan(user_id):
    response_schemas = [
        ResponseSchema(name="latest_login", description="The latest login of the user"),
        ResponseSchema(name="risk factor", description="The risk factor of the latest login of the user based on comparsion with previous logins. This could be low medium or high"),
        ResponseSchema(name="explanation", description="explanation of the risk factor of the latest login of the user"),
        ResponseSchema(name="message-if-high", description="If the risk factor is high, Draft a message to the user explaining what contributed to the risk factor and ask them to verify their account using multi factor authentication. if risk factor is medium or low, this should be Null"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    response_format = output_parser.get_format_instructions()
    pt = PromptTemplate(
            input_variables=["data"],
            partial_variables= {"response_format": response_format},
            template ="""
            You are a bot which finds anamolous user login activity. You are given user login logs which have 
            user_id,Name,Email,Age,Gender,Nationality,Time_of_Login ,Typing_Speed,Time_to_Complete_Captcha,Device,Local_IP,Global_IP,and OS. Print out the anamolous data points, explain why and display a risk factor for them. The risk factor is higher if the distance between the previous location and current location is high.
            ---
            {dataset}
            ---
            {response_format}
            """
            )
    chain1 = LLMChain(llm = llm, prompt=pt, 
                                    output_key="anamolies"
                                )
    out = chain1.run(dataset=df[df.user_id==user_id].to_string())
    print(output_parser.parse(out))

#scan(4)