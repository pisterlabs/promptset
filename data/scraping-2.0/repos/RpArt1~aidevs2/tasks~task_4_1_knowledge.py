from tasks.abstractTask import AbstractTask
from pydantic import BaseModel
from openai import OpenAI
import logging
from utils.assigment_utils import AssigmentUtils 
import json


class KnowledgeTask(AbstractTask):
    
    EXCHANGE_RATE_API = "https://api.nbp.pl/api/exchangerates/rates/A/{}"
    COUNTRY_API = "https://restcountries.com/v3.1/name/{}"
    
    def __init__(self, task_signature: str, send_to_aidevs: bool, mock: bool):
        super().__init__(task_signature, send_to_aidevs, mock)
        self.OPEN_AI_CLINET = OpenAI()

    
    def solve_task(self):
        return super().solve_task()
    
    def process_task_details(self):
        question = self.assignment_body["question"]
        logging.info(f"Question: {question}")
        
        if self.mock:
            ai_response_tuple = ('currency_name', 'KRW')
        else:
            ai_response_tuple = self.clasify_question(question)
        
        try: 
            if ai_response_tuple[0] == "country_name":
                self.process_population_question(ai_response_tuple[1]['country'])
            elif ai_response_tuple[0] == "currency_name":
                return self.process_currency_question(ai_response_tuple[1]['currency'])
            else:
                return ai_response_tuple[1]
        except Exception as e:
            logging.error(f"Error while processing question: {question} with error: {e}")
            return None
        
    def process_currency_question(self, currency: str) -> int:
        """
            Process question about population number
        """
        currency = currency.upper()
        url = self.EXCHANGE_RATE_API.format(currency)
        response = AssigmentUtils.process_request(url)
        if response.status_code == 200 and isinstance(response.json()['rates'][0]['mid'],float): 
            return response.json()['rates'][0]['mid']
        else:
            raise Exception(f"Error while fetching data from {url}")
    
    
    def process_population_question(self, country: str) -> int:
        """
            Process question about population number
        """
        country_url =self.COUNTRY_API.format(country.lower())
        logging.info(f"Fetching data from {country_url}")
        response = AssigmentUtils.process_request(country_url)
        if response.status_code == 200 and isinstance(response.json()[0]['population'],int): 
            return response.json()[0]['population']
        else:
            raise Exception(f"Error while fetching data from {country_url}")
    
    def clasify_question(self, question: str):
        """
            Classify question to one of two available openai functions.
            If none  choosen - question is general and will be answered by openai => ('', response)
            Otherwise wil return function name and essetial part of question - like country name or currency name => ('currency_name', 'Euro')
        """
        
        class CurrencySchema(BaseModel):
            currency: str 
            
        class CountrySchema(BaseModel):
            country: str
        
        
        currency_schema = CurrencySchema.model_json_schema()
        country_schema = CountrySchema.model_json_schema()
        
        currency_prompt = '''If user question is about currency return 3 letter code which follows 
            standard ISO 4217 and absolutely nothing else\n

            #### example:\n
            user: "What is current exchange rate for euro?"\n
            chat: EUR  \n\n
            #### example:\n
            user:"jaki jest prelicznik dla dolar australijskiego?"\n
            chat: AUD\n
        '''
        
        population_prompt = '''If the question is about population number, return country name in English language. Add nothing else \n
            Use ENglish language ONLY\n
            #### examples\n
            user: Ile ludzi jest w Stanach ?\n
            chat: USA\n\n
            user: Ile mniej więcej jest osób Na Tajwanie\n
            chat: Taiwan\n
            user: Podja mi populacje Niemiec\n
            chat: Germany\n
            user: Ile żyje osób Polsce ? \n
            chat: Poland\n
        '''
        
        function_list = [
            {
                "name": "currency_name",
                "description": currency_prompt,
                "parameters": currency_schema 
            },
            {
                "name": "country_name",
                "description": population_prompt,
                "parameters": country_schema
            }
        ]
        
        #debug question 
        #question = "What is the most important city in Poland?"
        #question = "What is people of Poland?"
        #question = "ile orienta cyjnie ludzi mieszka w Polsce?"
        response = self.OPEN_AI_CLINET.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
            {"role": "user", "content": question},
            ],
            functions=function_list,
        )
        
        function_call = response.choices[0].message.function_call
        if (function_call != None):
            function_call_name = function_call.name 
            logging.info(f"Function choosen: {function_call_name} for question: {question}")
            return function_call_name, json.loads(function_call.arguments)
        else: 
            logging.info(f"No function choosen - question clasifed as general: {question}")
            return '', response.choices[0].message.content