from tasks.abstractTask import AbstractTask
from openai import OpenAI
import logging
from pydantic import BaseModel
from typing import Literal
import json

class ToolsTask(AbstractTask):

    def __init__(self, task_signature: str, send_to_aidevs: bool, mock: bool):
        super().__init__(task_signature, send_to_aidevs, mock)
        self.OPEN_AI_CLINET = OpenAI()
        self.logger = logging.getLogger(__name__)

    
    def solve_task(self):
        return super().solve_task()
    
    def process_task_details(self):
        question = self.assignment_body["question"]
        return self.clasify_question(question)
    
    def clasify_question(self, question: str):
        """
            Classify question to one of 2 available openai answers.
            If none  choosen - nothing will be returned
            Otherwise wil return function name and essetial part of question - like date , command 
        """
        
        class EventSchema(BaseModel):
            tool: Literal['Calendar', 'ToDo']
            desc: str = ...
            date: str
        
        event_schema = EventSchema.model_json_schema()
        
        function_call_description = '''Clasify user input into 2 possible choices - Calendar or ToDo.
        ### Facts\n
        today is 2023-12-23\n
        ### Rules\n
        Choose calndar if user input contains command and date or day description - like tommorow\n
        Choose todo if user input contains command\n
        '''
        
        function_list = [
            {
                "name": "describe_event",
                "description": function_call_description,
                "parameters": event_schema 
            }
        ]
        
        #debug question 
        #question = "Pojutrze mam kupić 1kg ziemniaków"
        if (self.mock):
            self.logger.info(f"Mock mode - question clasifed as general: {question}")
            function_call = None
        else:    
            response = self.OPEN_AI_CLINET.chat.completions.create(
                model="gpt-4",
                temperature=0,
                messages=[
                {"role": "user", "content": question},
                ],
                functions=function_list,
            )
            function_call = response.choices[0].message.function_call
            
        if (function_call != None):
            function_call_name = function_call.name 
            self.logger.info(f"Function choosen: {function_call_name}, response: {function_call.arguments} for question: {question}")
            return  json.loads(function_call.arguments)
        else: 
            self.logger.info(f"No function choosen - question clasifed as general: {question}")
            return None