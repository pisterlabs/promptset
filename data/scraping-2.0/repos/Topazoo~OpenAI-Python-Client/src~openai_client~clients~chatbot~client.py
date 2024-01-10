import openai
from ...base import OpenAI_Client
from ...mixins import Chat_Context_Manager_Mixin
from ...enums import ROLE, CHAT_MODELS

# Typing
from typing import List, Dict


class Chat_Bot_Client(OpenAI_Client, Chat_Context_Manager_Mixin):
    _model = CHAT_MODELS.GPT_3_5_TURBO
    _api = openai.ChatCompletion
    _supported_models = CHAT_MODELS

    def __init__(self, max_stored_statements:int=-1, directives:List[str]=None, statements:List[Dict]=None, api_key:str=None, defaul_model_name:str="", default_temperature:float=0, max_retries:int=3, ms_between_retries:int=500) -> None:
        super().__init__(api_key, defaul_model_name, default_temperature, max_retries, ms_between_retries)
        Chat_Context_Manager_Mixin.__init__(self, max_stored_statements=max_stored_statements, directives=directives, statements=statements)

    def run_prompt(self, temperature: float = 0):
        """ Sends a prompt to OpenAI """

        # Call the API and get a response
        result = self._api.create(model=self._model, messages=self.get_context(), temperature=temperature or self._temperature)

        try:
            # If valid, save it in context to be passed on future requests
            response = result["choices"][0]["message"]["content"]
            if response:
                self.add_statement(ROLE.ASSISTANT, response)
                return response
            
        except Exception:
            raise Exception("Failed to get a response from OpenAI API")
        

    def get_user_input(self):
        """ Get user input to send to the chatbot, save for future requests """

        # Get a question from the user and store is
        user_input = input('>>> ')
        
        self.add_statement(ROLE.USER, user_input)
