from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
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

# Prompt Template for Intent Extraction
prompt_text_intent = """
You are a highly specialized AI trained to understand mortgage guidelines, terms, and policies.
Your task is to read a FAQ question and quickly identify the primary intent behind it. 
Please provide the intent as a single word or a short phrase without any explanations.

summarized_text: {input}

What is the primary intent?

"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str = Field(..., description="The template text for the prompt")

    def __init__(self, template, input_variables, tools=[]):
        super().__init__(input_variables=input_variables, tools=tools)
        self.template = template  # Initialize the instance attribute

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

# Create an instance of CustomPromptTemplate for Intent
prompt_intent = CustomPromptTemplate(template=prompt_text_intent, input_variables=["input"], tools=[])

# Output Parser for Intents
class IntentOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        intent_identified = llm_output.strip()
        return AgentFinish(
            return_values={
                "intent": intent_identified,
                "output": llm_output
            },
            log={"llm_output": llm_output}
        )

intent_output_parser = IntentOutputParser()

# Set up LLM for FAQ generation
llm = ChatOpenAI(model_name="ft:gpt-3.5-turbo-0613:personal::7scg7esv", temperature=0.9)

# LLM Chain
llm_chain = LLMChain(llm=llm, prompt=prompt_intent)

# Set up the Agent for Intent extraction
intent_agent = LLMSingleActionAgent(
    llm_chain=LLMChain(llm=llm, prompt=prompt_intent),
    output_parser=intent_output_parser,
    stop=["\n"]
)

# Agent Executor for Intent
intent_agent_executor = AgentExecutor(agent=intent_agent, tools=[])

# Initialize a DataFrame to store the summarized text, if 'summarized_text.csv' exists, it will load it
try:
    df_intent = pd.read_csv("Data/Extracted_Data/master_with_intent.csv")
except FileNotFoundError:
    df_intent = pd.DataFrame()

# Read your input CSV file
input_csv_path = "Data/Extracted_Data/master_with_faq.csv"
df = pd.read_csv(input_csv_path)

# Initialize an empty list to store extracted intents
extracted_intents = []

# Extract intents based on the `summarized_text` column of the DataFrame
for index, row in df.iterrows():
    summarized_text = row['summarized_text']
    intent_response = intent_agent_executor.run({"input": summarized_text})
    # Extract the intent from the response
    extracted_intent = intent_response.replace("intent: ", "").strip()
    extracted_intents.append(extracted_intent)

# Add the extracted intents to the DataFrame
df['Intent'] = extracted_intents

# Append to the existing DataFrame for summarized text
df_intent = pd.concat([df_intent, df], ignore_index=True)

# Save the DataFrame
df_intent.to_csv("Data/Extracted_Data/master_with_intent.csv", index=False)

