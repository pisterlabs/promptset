from tasks.abstractTask import AbstractTask
from utils.assigment_utils import AssigmentUtils
from openai import OpenAI
import json
import ast
import logging


class PeopleTask(AbstractTask):
    
    URL = "https://zadania.aidevs.pl/data/people.json"
    
    def __init__(self, task_name, send_to_aidevs, mock):
        super().__init__(task_name, send_to_aidevs, mock)
        self.client = OpenAI()
        # self.logger = project_logger
        
        
    def solve_task(self):
        logging.debug('Debug message from MyClass')
        logging.info('And this is info')
        logging.warn('And this is warn')
        logging.error('And this is error')
        
        

        return super().solve_task()
    
    def process_task_details(self):
        people = AssigmentUtils.process_request(self.URL).json()
        question = self.assignment_body['question']
        
        # search entry based on question in people list
        # search entry based on name and surname fetched from question in people list
        # how to extract name and surname from question? → regex or llm ? 
        name_and_surname = self.extract_name_and_surname(question)
        index = {(entry['imie'], entry['nazwisko']): entry for entry in people}

        found_entry = index.get(name_and_surname)
        answer_question = self.answer_question(question, found_entry)
        
        return answer_question

    def answer_question(self, question : str, found_entry : dict):
        """
            Answer question based on found entry using AI 
        
        """
        
        
        found_entry_str = json.dumps(found_entry)
        
        system_prompt = f''' Anwer only based on below data as short as possible: \n
        {found_entry_str} \n
        ### example 1: \n
        user: "jaki jest ulubion pies Marka ? " \n
        chat: "Rex"\n

        ### example 1:\n
        user: "W którym roku był Andrzej na wyborach?" \n
        chat: 1933\n
        ''' 
        
        if self.mock:
            question = 'Ulubiony kolor Agnieszki Rozkaz, to?'
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}

            ]
        )
        text_response =  response.choices[0].message.content
        return text_response
                
        
        
    
    def extract_name_and_surname(self, question : str) :
        
                
        """
            Extract name and surname from question using regex
            If mock enabled returns mock response
            
            return tuple (name, surname)

        """
        
        if self.mock:
            return ("Agnieszka", "Rozkaz")

        system_prompt = ''' 
        Don't answer user qustions. Fetch from user qusion only name and surname and return in format as in below examples: \n  
        ## example 1 \n
        user: Jaki jest ulubiony psiak Jana Kowalskiego ? \n 
        chat:  ( "Jan", "Kowalski")  \n 
        ## example 2 \n
        user: Anna Chałas pisze zejbiste książki \n
        chat: (Anna", "Chałas") \n
        '''
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}

            ]
        )
        text_response =  response.choices[0].message.content
        try:
            return ast.literal_eval(text_response)
        
        except Exception as e:
            # self.logger.error(f"Could not convert str {text_response} into tuple")
            # self.logger.debug(f"Exception: {e}")
            return None
    
        


