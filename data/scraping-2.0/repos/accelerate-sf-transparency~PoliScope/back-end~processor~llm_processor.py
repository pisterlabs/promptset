import os
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationEntityMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.chains import LLMChain

import pandas as pd
import numpy as np


class LLMProcessor:
    def __init__(self, data):
        self.data = pd.read_csv(data) ## Testing on 5 rows
        self.data = self.data.dropna(how='any')

    def initialize_agent(self, agent_def_path, model_name="gpt-3.5-turbo"):
        """Initialize the agent and return it."""
        ## Load prompts
        with open(agent_def_path) as f:
            system_prompt = f.read()
        
        # Setup the chat prompt with memory
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),  # The persistent system prompt
            # MessagesPlaceholder(variable_name="chat_history"),  # Where the memory will be stored
            HumanMessagePromptTemplate.from_template("{human_input}"),  # Human input
        ])

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            request_timeout=120,
        )

        # memory = ConversationEntityMemory(llm=llm)

        chat_llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )

        self.agent = chat_llm_chain
        return self.agent
    
    def custom_function(self, row):
        # Map of category to score column
        category_column_map = {
            'Affordable Housing Development': 'affordable_housing_development_score',
            'Tenant Protections': 'tenant_protections_score',
            'Homelessness and Supportive Housing': 'homelessness_and_supportive_housing_score',
            'Permitting Process and Bureaucratic Efficiency': 'faster_permitting_process_and_bureaucracy_score',
            'Land Use and Zoning Reforms': 'land_use_and_zoning_reform_score'
        }
        # Get the score column corresponding to the category
        score_column = category_column_map.get(row['category'])
        if score_column:
            # If the category matches, retain the score; else set to None
            row[score_column] = row['score']
        return row

    def conditionally_set_null(self):
        # Define the custom function to apply
        
        # Set all score columns to None initially
        self.data[['affordable_housing_development_score', 'tenant_protections_score',
                   'homelessness_and_supportive_housing_score', 'faster_permitting_process_and_bureaucracy_score',
                   'land_use_and_zoning_reform_score']] = None
        
        # Use DataFrame.apply with axis=1 to apply the custom function row-wise
        self.data = self.data.apply(self.custom_function, axis=1)
        self.data.drop(['score'], inplace=True)
        return self.data

    def run_agent(self, agent, human_input):
        """Run the provided agent using the input and return the result."""
        result = agent.predict(human_input=human_input)
        return result

    def apply_summarizer_agent(self, row):
        return self.run_agent(self.summarizer_agent, row['Title'])
    
    def apply_categorizer_agent(self, row):
        return self.run_agent(self.categorizer_agent, row['Title'])

    def apply_categorizer_big_agent(self, row):
        return self.run_agent(self.categorizer_big_agent, row['Title'])

    def apply_scorer_agent(self, row):
        return self.run_agent(self.scorer_agent, row['scorer_field'])
    
    def apply_positions_agent(self, row):
        return self.run_agent(self.positions_agent, row["position"])
    
    def process_positions(self):
        data_path_name = './data/csv/positions.csv'
        data = pd.read_csv(data_path_name)
        # llm_response = self.run_agent(self.agent, self.data['Title'][1])
        data['position'] = data.apply(self.apply_positions_agent, axis=1)
        return self.data

    def process(self):

        # Run LLM on combined fields and create summary table
        # self.data['category_big'] = self.data.apply(self.apply_categorizer_big_agent, axis=1)
        # self.data = self.data[self.data['category_big'] =='Housing & Buildings']

        # llm_response = self.run_agent(self.agent, self.data['Title'][1])
        self.data['category'] = self.data.apply(self.apply_categorizer_agent, axis=1)
        self.data['summary'] = self.data.apply(self.apply_summarizer_agent, axis=1)
        # print(llm_response)
        
        self.data['scorer_field'] = '"' + self.data['category'] + '"' +  ', ' + '"' + self.data['Vote'] + '"' + ', ' +  '"' + self.data['Title'] +  '"'

        ## Apply on all not in 'other' category field
        self.data['score'] = self.data.apply(self.apply_scorer_agent, axis=1)
        self.conditionally_set_null()

        self.data = self.data.fillna(value=0.0)

        return self.data

if __name__=='__main__':

    test_processor = LLMProcessor('../data/ingest.csv')
    test_processor.initialize_agent('./agent_prompts/summarizer.txt')

    test_processor.process()


