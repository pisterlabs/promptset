import openai
import logging
import re
import json
import os
import yaml
from injector import Injector

from src.container_config import container
from src.config_manager import ConfigManager
from src.response_handler import FileResponseHandler
from src.source_code_response_handler import SourceCodeResponseHandler
from src.agent_manager import AgentManager
from src.message_processors.message_preProcess import MessagePreProcess
from src.api_helpers import ask_question, get_completion, get_completionWithFunctions
from src.message_processors.message_processor_interface import MessageProcessorInterface
from src.message_processors.function_calling_processor import FunctionCallingProcessor

from src.prompt_builders.prompt_builder import PromptBuilder
# from src.completion.completion_manager import CompletionManager
from src.completion.completion_store import CompletionStore
from src.handlers.quokka_loki import QuokkaLoki
from src.handlers.task_update_handler import TaskUpdateHandler
from src.handlers.file_save_handler import FileSaveHandler
from src.handlers.command_execution_handler import CommandExecutionHandler
from src.handlers.user_action_required_handler import UserActionRequiredHandler
from src.handlers.file_load_handler import FileLoadHandler
from src.handlers.web_search_handler import WebSearchHandler
from src.handlers.scrape_web_page_handler import ScrapeWebPage

from src.message_processors.message_processor import MessageProcessor
from task_generator import TaskGenerator
from src.context.context import Context
from src.context.context_manager import ContextManager
from src.hierarchical_node import HierarchicalNode
from src.node_manager import NodeManager
from src.presets.preset_handler import PresetHandler
from src.presets.preset_prompts import PresetPrompts
from src.task_creation.task_builder import TaskBuilder

config = ConfigManager('config.json')


class AutomationProcessor(MessageProcessorInterface):

    def __init__(self ):
        base_path = os.path.dirname(os.path.realpath('sandpit'))
        self.goal = None
        self.steps_yaml = None
        self.steps = None
        self.step_context = None
        self.goal_context = None
        self.node_manager = container.get(NodeManager)
        task_update_handler = TaskUpdateHandler(self.node_manager) 
        file_save_handler = FileSaveHandler()
        command_execution_handler = CommandExecutionHandler()
        user_action_required_handler = UserActionRequiredHandler()
        file_load_handler = FileLoadHandler()
        self.handler = QuokkaLoki()
        self.handler.add_handler(file_save_handler)
        self.handler.add_handler(command_execution_handler)
        #self.handler.add_handler(user_action_required_handler)
        self.handler.add_handler(file_load_handler)
        self.handler.add_handler(task_update_handler)
        self.handler.add_handler(WebSearchHandler())
        #self.task_generator = TaskGenerator(self.node_manager)
        self.presets = PresetPrompts(config.get('preset_path'))
        self.task_builder = TaskBuilder()
        self.context_mgr =ContextManager(config)
        self.task_prompt_part = self.presets.get_prompt('task_prompt_part')
        self.task_prompt_retry = self.presets.get_prompt('task_prompt_retry')
        self.task_prompt_work = self.presets.get_prompt('task_prompt_work')
        self.task_prompt_critic = self.presets.get_prompt('task_prompt_critic')
        

    def process_message(self, agent_name: str, account_name: str, message: str, conversationId="0", context_name='', second_agent_name='') -> str:
        logging.info(f'Function Calling Processing message inbound: {message}')
        response = ""
        agent_manager = container.get(AgentManager)
        agent = agent_manager.get_agent(agent_name)
        model = agent["model"]
        temperature = agent["temperature"]
        context_type = agent["select_type"]


        # set up the nodes for the steps in the process
        max_iterations = 10


        task_builder = TaskBuilder()
        task_builder.create_task_nodes(message, account_name, conversationId)

        self.top_node = self.node_manager.get_nodes_conversation_id(conversationId, "top")
        self.top_node = self.top_node[0]
        self.top_node.account_name = account_name

        self.task_nodes = self.node_manager.get_nodes_parent_id(self.top_node.id) 
        self.top_context = self.get_context_from_node(context_name, self.top_node)

        #iterate thought the steps
        itr = 0
        max_iterations = 10
        while itr < max_iterations:

            next_step = self.find_next_step(self.steps)
            if next_step:
                next_step.account_name = account_name
                self.process_step(next_step, agent_name, second_agent_name, account_name, conversationId, context_name)
            else:
                print("No more steps to process")
                break

            itr += 1

        automation_response = ''
        for step in self.task_nodes :
            automation_response += step.description + ' state: ' + step.state + '\n'

        return automation_response


    def process_step(self, step: HierarchicalNode, primary_agent_name:str, second_agent_name:str, account_name:str, conversationId:str, context_name:str):
        if step:

            if step.state == 'none':
                step.state = 'in_progress'
                # set up the context used between the agents when processing the step
                if self.step_context is None:
                    self.step_context = self.get_context_from_node(context_name, step)
                else:
                    self.step_context.description = step.description
                    self.step_context.name = context_name
                    self.step_context.state = step.state
                    self.step_context.current_node_id = step.id
                    self.step_context.info = step.info
                    self.step_context.output_directory = step.working_directory
                    self.step_context.input_directory = step.working_directory


            # extract the text version of the context this will be sent to the agent's compeltion API
            context_text = self.step_context.get_info_text()  

            my_step_text = step.get_formatted_text(["name", "description", "info", "state"])

            # set up the text that will be sent to the/completion API this basically consists of the the instruction for the agent (task_prompt) and the context
            #if(primary_agent_name == 'doug'):
            #    message = self.task_prompt_work['prompt'] + "  "  + 'context_info:'  + context_text + " "
            if step.state == 'retry':
                message = self.task_prompt_retry['prompt'] + "  "  + 'context_info:'  + context_text + " "
            else:
                #message = self.task_prompt_part['prompt'] + "  " + 'context_info:'  + context_text + " " 
                message = "hello your task is " + self.step_context.description + "  " + 'context_info:'  + context_text + " " 

            response = self.ask(message, primary_agent_name, account_name, conversationId, '')

            # running a step will most likely generate actions that need to be executed

            #self.step_context.actions = []  # clear actions  

            #rh_repsonse = self.handler.process_request(response)
            #response_text = QuokkaLoki.handler_repsonse_formated_text(rh_repsonse)


            #self.handler.account_name = account_name
            ###rh_repsonse = self.handler.process_request(response)
            # = QuokkaLoki.handler_repsonse_formated_text(rh_repsonse)
            #if response_text != '':
            #    response = response + " response: " + response_text
            #    if context_name != "" and context_name != "none":
            #        self.step_context.update_from_results(rh_repsonse)
            #else:
            #    self.step_context.add_action(step.name, '', 'Result' + response)
     
 
                
            # when any actions are executed we need to find out if they waorked as expected
            my_request_summary = step.get_formatted_text(["name", "description"])
            critic_prompt = self.task_prompt_critic['prompt'] + my_request_summary + " here is the response:" + response 
            critic_response = self.ask(critic_prompt, second_agent_name, account_name, conversationId,'')
            critic_response = critic_response.lower()
            if "yes" in critic_response:
                step.state = 'completed'
                #self.step_context.update_from_results(rh_repsonse)
                #self.step_context.description
                #self.step_context.add_action(response, ' ', 'SUCCESS')   
            else:
                step.state = 'retry'
                self.step_context.add_action(response, '', 'ERROR -' + critic_response)
                self.step_context.retry_information = critic_response

            self.context_mgr.post_context(self.step_context)
            print(self.step_context.get_info_text())


    

  

    def find_next_step(self, steps):
        # Check if there's an "in progress" step
        for node in self.task_nodes:
            if node.state == 'in_progress':
                print('A step is currently in progress.')
                return None

        # If there's no "in progress" step, find the next available step
        for node in self.task_nodes:
            if node.state == 'none':
                return node
            if node.state == 'retry':
                return node

        # If no next step is found, return None
        return None
    
    
    def get_context_from_node(self, context_name: str, node: HierarchicalNode) -> Context:
        context = Context(name=context_name, description=node.description, current_node_id=node.id, state=node.state,account_name=node.account_name, conversation_id=node.conversation_id)
        context.output_directory = node.working_directory
        context.input_directory = node.working_directory
        context.account_name = node.account_name
        context.add_info(node.info)
        self.context_mgr.post_context(context)

        return context


    def ask(self, question: str, agent_name:str , account_name:str, conversation_id:str, context_name) -> str:

        agent_manager = container.get(AgentManager)
        my_agent = agent_manager.get_agent(agent_name)

        partner_agent = ""
        if('partner_agent' in my_agent):
            partner_agent = my_agent['partner_agent']


        if 'message_processor' in my_agent and my_agent['message_processor'] == 'function_calling_processor':
            processor = FunctionCallingProcessor()
        else:
            processor = MessageProcessor()

        processor = FunctionCallingProcessor()

        # need to persist the context before and after processing the message
        self.context_mgr.post_context(self.step_context)
        response = processor.process_message(agent_name, account_name, question, conversation_id, context_name)
        if context_name != '':
            self.step_context = self.context_mgr.get_context(account_name, context_name)

        # processor.save_conversations()
        return response

    
