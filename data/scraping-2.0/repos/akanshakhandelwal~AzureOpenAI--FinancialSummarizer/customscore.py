
import os
from langchain.llms import OpenAI
import streamlit as st
import os
import langchain
import pypdf
import unstructured 
import utils
from langchain.document_loaders import MergedDataLoader
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import json
from pprint import pprint

PROMPTS = {
    'credit_ratings': "Extract credit ratings from the document of {}. Score 1-10 based on AAA=10 to D=1 scale. If there is no mention of credit rating, give it a score of 0",
    'debt_to_equity': "Calculate the debt-to-equity ratio from the balance sheet in document of {}. Score: <0.5=10, 0.5-1=8, 1-1.5=6, >1.5=4.",
    'interest_coverage': "Calculate the interest coverage ratio from the financials in document of {}. Score: >5=10, 3-5=7, 1-3=4, <1=2. ",
    'liquidity_ratio': "Calculate the liquidity ratio in document for {}. Score: >2=10, 1.5-2=8, 1-1.5=6, <1=4.",
    'profit_margin': "Calculate the profit margin in document for {}. Score: >20%=10, 15-20%=8, 10-15%=6, <10%=4.",
    'revenue_growth': "Calculate the revenue growth rate in document for {}. Score: >15%=10, 10-15%=8, 5-10%=6, <5%=4.",
    'management_quality': "Assess the management quality in document of {}. Score: Excellent=10, Good=8, Average=6, Poor=4.",
    'legal_compliance': "Assess the legal compliance of {} based on pdocument. Score: Excellent=10, Good=8, Average=6, Poor=4."
}
class RiskScore:
    def __init__(self):
        self.total_score = 0
        self.total_weight = 0

    def api_call(self, prompt,company_name):

        vector_store=utils.get_azure_vector_store()
        prompt = prompt.format(company_name)
        response = utils.ask_and_get_answer(vector_store, prompt)
       
        
        
        result_str = response['result']
        
        # Replace single quotes with double quotes to make it a valid JSON string
        # result_str = result_str.replace("'", '"')
        print(result_str)
        # Convert the JSON string to a Python dictionary
        result_dict = json.loads(result_str)
        print(result_str)
        # Extract the score from the dictionary
        score = result_dict['score']
       
        explanation = result_dict['explanation']
        # pprint(f"Score: {score}, Type: {type(score)}") 
        # pprint(f"Score: {explanation}") 
        return score,explanation

    def credit_ratings(self, company_name,weight):
        prompt = PROMPTS["credit_ratings"]
        score,explanation = self.api_call(prompt,company_name)
        self.total_score += score * (weight if score != -1 else 0)
        self.total_weight += weight if score != -1 else 0
        return score,explanation

    def debt_to_equity(self, company_name,weight):
        prompt = PROMPTS["debt_to_equity"]
        score,explanation = self.api_call(prompt,company_name)
        self.total_score += score * (weight if score != -1 else 0)
        self.total_weight += weight if score != -1 else 0
        return score,explanation
        

    def interest_coverage(self, company_name,weight):
        prompt = PROMPTS["interest_coverage"]
        score,explanation = self.api_call(prompt,company_name)
        self.total_score += score * (weight if score != -1 else 0)
        self.total_weight += weight if score != -1 else 0
        return score,explanation

    def liquidity_ratio(self,company_name, weight):
        prompt = PROMPTS["liquidity_ratio"]
        score,explanation = self.api_call(prompt,company_name)
        self.total_score += score * (weight if score != -1 else 0)
        self.total_weight += weight if score != -1 else 0
        return score,explanation

    def profit_margin(self,company_name, weight):
        prompt = PROMPTS["profit_margin"]
        score,explanation = self.api_call(prompt,company_name)
        self.total_score += score * (weight if score != -1 else 0)
        self.total_weight += weight if score != -1 else 0
        return score,explanation

    def revenue_growth(self, company_name,weight):
        prompt = PROMPTS["revenue_growth"]
        score,explanation = self.api_call(prompt,company_name)
        self.total_score += score * (weight if score != -1 else 0)
        self.total_weight += weight if score != -1 else 0
        return score,explanation

    def legal_compliance(self, company_name,weight):
        prompt = PROMPTS["legal_compliance"]
        score,explanation = self.api_call(prompt,company_name)
        self.total_score += score * (weight if score != -1 else 0)
        self.total_weight += weight if score != -1 else 0
        return score,explanation

    def management_quality(self, company_name,weight):
        prompt = PROMPTS["management_quality"]
        score,explanation = self.api_call(prompt,company_name)
        self.total_score += score * (weight if score != -1 else 0)
        self.total_weight += weight if score != -1 else 0
        return score,explanation

    

    def calculate_overall_risk_score(self):
        if self.total_weight == 0:
            return -1  # Handling the case where all weights are 0
        return self.total_score / self.total_weight
