import openai
import logging
import re
from injector import Injector

from src.container_config import container
from src.config_manager import ConfigManager  

from src.agent_manager import AgentManager
from src.message_processors.message_preProcess import MessagePreProcess
from src.api_helpers import ask_question, get_completion
from src.message_processors.message_processor_interface import MessageProcessorInterface

from src.prompt_builders.prompt_builder import PromptBuilder

from src.completion.completion_store import CompletionStore


from src.handlers.quokka_loki import QuokkaLoki

from src.handlers.file_save_handler import FileSaveHandler
from src.handlers.command_execution_handler import CommandExecutionHandler  

from src.handlers.file_load_handler import FileLoadHandler
from src.handlers.web_search_handler import WebSearchHandler
from src.context.context_manager import ContextManager
from src.context.context import Context


class MessageProcessor(MessageProcessorInterface):
    def __init__(self ):
        # self.name = name
        agent_manager = container.get(AgentManager)
        self.config = container.get(ConfigManager) 
        prompt_base_path=self.config.get('prompt_base_path')  
        self.seed_conversations = []
        

        self.handler = QuokkaLoki()
        self.handler.add_handler(FileSaveHandler())
        self.handler.add_handler(CommandExecutionHandler())
        self.handler.add_handler(FileLoadHandler())
        self.handler.add_handler(WebSearchHandler())
        


    def process_message(self, agent_name:str, account_name:str, message, conversationId="0", context_name='', second_agent_name='') -> str:
        logging.info(f'Processing message inbound: {message}')
        agent_manager = container.get(AgentManager)
        agent = agent_manager.get_agent(agent_name)
        model = agent["model"]
        temperature = agent["temperature"]
        context_type = agent["select_type"]

        completion_manager_store = container.get(CompletionStore)
        account_completion_manager = completion_manager_store.get_completion_manager(agent_name, account_name, agent['language_code'][:2])


        # Check for alternative processing
        preprocessor = MessagePreProcess()
        myResult = preprocessor.alternative_processing(message, conversationId, agent_name, account_name)

        if myResult["action"] == "return":
            return myResult["result"]
        if myResult["action"] == "storereturn":
            self.add_response_message(conversationId,  message, myResult["result"])
            return myResult["result"]
        elif myResult["action"] == "continue":
             message = myResult["result"]
        elif myResult["action"] == "swapseed":
             seed_info = myResult["result"]
             seed_name=seed_info["seedName"]
             seed_paramters=seed_info["values"]
        elif myResult["action"] == "updatemessage":
             message =  myResult["result"]
 

        prompt_builder = PromptBuilder()
        conversation = prompt_builder.build_prompt( message, conversationId, agent_name, account_name, context_type, 6000, 20, context_name)
       
       
        # coversations [ {role: user, content: message} ]
        logging.info(f'Processing message prompt: {conversation}')

        response = ask_question(conversation, model, temperature)  #string response

        logging.info(f'Processing message response: {response}')
        self.post_process = False   # disabled for now - see automation_processor
        if self.post_process :
            self.handler.account_name = account_name
            rh_repsonse = self.handler.process_request(response)
            response_text = QuokkaLoki.handler_repsonse_formated_text(rh_repsonse)
            if response_text != '':
                response = response + " response: " + response_text
                if context_name != "" and context_name != "none":
                    context_mgr =ContextManager(self.config)
                    context = context_mgr.get_context(account_name, context_name)
                    context.update_from_results(rh_repsonse)
                    context_mgr.post_context(context)
                
        

        if agent['save_reposnses']:
            if message is not self.is_none_or_empty(message):
                account_completion_manager.create_store_completion(conversationId, message, response)
                account_completion_manager.save()

        logging.info(f'Processing message complete: {message}')

        return response
    
    def is_none_or_empty(self, string):
        return string is None or string.strip() == ""
    



    
    

        

    
