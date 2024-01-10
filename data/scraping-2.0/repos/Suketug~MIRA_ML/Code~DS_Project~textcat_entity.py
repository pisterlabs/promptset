from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import pandas as pd
import json
import os
import openai
import dotenv
from pydantic import Field

# Load the .env file
dotenv.load_dotenv()

# Initialize OpenAI
API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY

prompt_entity_extraction = """
You are a specialized AI trained in mortgage lending guidelines, terms, and policies.
Given a summarized description of texts, your task is to Extract the most important ENTITIES related to mortgage lending from the summarized_text column.
You are to read through each cell and extract entities from each cell in summarized_text column provided. you do not need to provide explanation for the same. 

Text: {input}
"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str = Field(..., description="The template text for the prompt")

    def __init__(self, template, input_variables, tools=[]):
        super().__init__(input_variables=input_variables, tools=tools)
        self.template = template  # Initialize the instance attribute

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

# New Output Parser for Entity Extraction
class EntityExtractionOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> AgentFinish:
        entities = llm_output.strip()
        
        return AgentFinish(
            return_values={
                "entities": entities,
                "output": llm_output
            },
            log={"entities": entities}
        )

# Create an instance of CustomPromptTemplate
prompt = CustomPromptTemplate(template=prompt_entity_extraction, input_variables=["input"], tools=[])

# Create an instance of EntityExtractionOutputParser
output_parser = EntityExtractionOutputParser()

# Set up LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0.9)

# LLM Chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Set up the Agent
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\n"]
)

# Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=[])

# Initialize a DataFrame to store the entities
try:
    df_extracted = pd.read_csv("Data/Full_Extraction/1_ext_entities.csv")
except FileNotFoundError:
    df_extracted = pd.DataFrame()

# Read the input CSV
input_csv_path = "Data/Full_Extraction/missing_ent.csv"
df = pd.read_csv(input_csv_path)

# Initialize empty list to store the entities
entities_list = []

# Iterate through the DataFrame
for index, row in df.iterrows():
    summarized_text = row['Cleaned_Description']
    raw_response = agent_executor.run({"input": summarized_text})
    
    # Parse the raw_response string here
    entities = raw_response.strip()
    
    # Append to list
    entities_list.append(entities)

# Add the entities to the DataFrame
df['entities'] = entities_list

# Append to the existing DataFrame for extracted entities
df_extracted = pd.concat([df_extracted, df], ignore_index=True)

# Save the updated DataFrame back to the same CSV file
df_extracted.to_csv("Data/Full_Extraction/1_ext_entities.csv", index=False)
