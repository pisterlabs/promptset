# Copyright (C) 2023  The CivRealm project
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import random
import json
import time

import openai
import pinecone

from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI, AzureOpenAI
from langchain.chains.question_answering import load_qa_chain

from civrealm.freeciv.utils.freeciv_logging import fc_logger
from agents.prompt_handlers.base_prompt_handler import BasePromptHandler

from .base_worker import BaseWorker


class AzureGPTWorker(BaseWorker):
    """
    This agent uses GPT-3 to generate actions.
    """
    def __init__(self,
                 model: str = 'gpt-35-turbo-16k',
                 prompt_prefix: str = "civ_prompts",
                 **kwargs):
        assert os.environ['OPENAI_API_TYPE'] == 'azure'
        self.prompt_prefix = prompt_prefix
        super().__init__(model, **kwargs)

    def init_prompts(self):
        self.prompt_handler = BasePromptHandler(
            prompt_prefix=self.prompt_prefix)
        self._load_instruction_prompt()
        self._load_task_prompt()

    def init_llm(self):
        openai.api_type = os.environ["OPENAI_API_TYPE"]
        openai.api_version = os.environ["OPENAI_API_VERSION"]
        openai.api_base = os.environ["OPENAI_API_BASE"]
        openai.api_key = os.environ["OPENAI_API_KEY"]

        self.deployment_name = os.environ['DEPLOYMENT_NAME']
        llm = AzureChatOpenAI(openai_api_base=openai.api_base,
                              openai_api_version=openai.api_version,
                              openai_api_key=openai.api_key,
                              openai_api_type=openai.api_type,
                              deployment_name=self.deployment_name,
                              temperature=0.7)
        self.chain = load_qa_chain(AzureOpenAI(
            deployment_name=self.deployment_name, model_name=self.model),
                                   chain_type="stuff")
        self.memory = ConversationSummaryBufferMemory(llm=llm,
                                                      max_token_limit=500)

    def init_index(self):
        pinecone.init(api_key=os.environ["MY_PINECONE_API_KEY"],
                      environment=os.environ["MY_PINECONE_ENV"])
        # print(os.environ)
        self.index = Pinecone.from_existing_index(
            index_name='civrealm-mastaba',
            embedding=OpenAIEmbeddings(model="text-embedding-ada-002"))

    def _load_instruction_prompt(self):
        instruction_prompt = self.prompt_handler.instruction_prompt()
        self.add_user_message_to_dialogue(instruction_prompt)

    def _load_task_prompt(self):
        task_prompt = self.prompt_handler.task_prompt()
        self.add_user_message_to_dialogue(task_prompt)

    def register_all_commands(self):
        self.register_command('manualAndHistorySearch',
                              self.handle_command_manual_and_history_search)
        self.register_command('finalDecision',
                              self.handle_command_final_decision)
        self.register_command('suggestion', self.handle_command_suggestion)

    def handle_command_manual_and_history_search(self, command_input,
                                                 obs_input_prompt,
                                                 current_avail_actions):
        if self.taken_actions_list_needs_update('look_up', 0, 0):
            answer = self.prompt_handler.finish_look_for()
            self.add_user_message_to_dialogue(answer)
            return None, ''

        query = command_input['look_up']
        answer = self.get_answer_from_index(query) + '.\n'
        if random.random() > 0.5:
            answer += self.prompt_handler.finish_look_for()

        self.add_user_message_to_dialogue(answer)
        self.memory.save_context({'assistant': query}, {'user': answer})
        self.taken_actions_list.append('look_up')
        return None, ''

    def handle_command_ask_current_game_information(self, command_input,
                                                    obs_input_prompt,
                                                    current_avail_actions):
        self.taken_actions_list.append('askCurrentGameInformation')
        return None, ''

    def handle_command_suggestion(self, command_input, obs_input_prompt,
                                  current_avail_actions):
        exec_action = command_input["suggestion"]
        return exec_action, ''

    def handle_command_final_decision(self, command_input, obs_input_prompt,
                                      current_avail_actions):
        exec_action = command_input['action']
        lower_avail_actions = [x.lower() for x in current_avail_actions]
        if exec_action.lower() not in lower_avail_actions:
            print(f'{self.name}\'s chosen action "{exec_action}" not in the ' +
                  f'available action list, available actions are ' +
                  f'{current_avail_actions}, retrying...')
            fc_logger.error(
                f'{self.name}\'s chosen action "{exec_action}"',
                'not in the available action list, available',
                f'actions are {current_avail_actions}, retrying...')
            return None, self.prompt_handler.insist_avail_action()

        self.taken_actions_list.append(command_input['action'])

        for move_name in current_avail_actions:
            if move_name[:4] != "move":
                continue
            if self.taken_actions_list_needs_update(move_name, 15, 4):
                return None, self.prompt_handler.insist_various_actions(
                    action=move_name)

        return exec_action, ''

    def query_llm(self, stop=None, temperature=0.7, top_p=0.95):
        fc_logger.debug(f'Querying with dialogue: {self.dialogue}')
        assert openai.api_type == 'azure'

        return openai.ChatCompletion.create(deployment_id=self.deployment_name,
                                            model=self.model,
                                            messages=self.dialogue,
                                            request_timeout=10)

    def generate_command(self, prompt: str):
        self.add_user_message_to_dialogue(prompt +
                                          self.prompt_handler.insist_json())
        self.restrict_dialogue()
        response = self.query_llm()
        self.memory.save_context({'user': prompt},
                                 {'assistant': str(response)})
        return response

    def parse_response(self, response):
        content = response['choices'][0]['message']['content']
        start_index = content.find('{')
        end_index = content.rfind('}') + 1
        rlack = content.count("{") - content.count("}")
        if rlack > 0:
            content = content[start_index:end_index] + "}" * rlack
        return json.loads(content)

    def process_command(self, response, obs_input_prompt,
                        current_avail_actions):
        # First try to parse the reponse by the given json format
        fc_logger.debug(f'Processing response: {response}')
        try:
            command_json = self.parse_response(response)
            command_input = command_json['command']['input']
            command_name = command_json['command']['name']
        except Exception as e:
            fc_logger.error(
                f'\nRESPONSE:{response}\nCommond json parsing error: {e}')
            print('Not in given json format, retrying...')
            return None, self.prompt_handler.insist_json()

        # Then check if the command is valid
        if command_name not in self.command_handlers:
            fc_logger.error(f'Unknown command: {command_name}')
            available_commands = ', '.join(self.command_handlers.keys())
            prompt_addition = self.prompt_handler.insist_available_commands(
                available_commands)
            return None, prompt_addition

        return self.command_handlers[command_name](command_input,
                                                   obs_input_prompt,
                                                   current_avail_actions)
