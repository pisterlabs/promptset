import os
import openai
import time
import random
import json
import requests
import warnings

from civrealm.freeciv.utils.freeciv_logging import fc_logger
from .utils import num_tokens_from_messages, send_message_to_llama, send_message_to_vicuna, extract_json, send_message_to_llama, TOKEN_LIMIT_TABLE
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI, AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
from agents.prompt_handlers.base_prompt_handler import BasePromptHandler

warnings.filterwarnings('ignore')

cwd = os.getcwd()
OPENAI_KEYS_FILE = os.path.join(cwd, "agents/civ_autogpt/openai_keys.txt")
SAVED_DIALOGUE_FILE = os.path.join(
    cwd, "agents/civ_autogpt/saved_dialogues/saved_dialogue.txt")
task_prompt_file = os.path.join(cwd,
                                "agents/civ_autogpt/prompts/task_prompt.txt")
GPT_MODEL_NAME = "gpt-3.5-turbo-16k"


class GPTAgent:
    """
    This agent uses GPT-3 to generate actions.
    """
    def __init__(self, model):
        self.model = model
        self.dialogue = []
        self.taken_actions_list = []
        self.message = ''

        self.openai_api_keys = self.load_openai_keys()
        self.prompt_handler = BasePromptHandler()
        self.state_prompt = self._load_state_prompt()
        self.task_prompt = self._load_task_prompt()

        # For Azure OpenAI API only
        self.deployment_name = os.environ['DEPLOYMENT_NAME']

        if os.environ['OPENAI_API_TYPE'] == 'azure':
            self.change_api_base('azure')
            llm = AzureChatOpenAI(openai_api_base=openai.api_base,
                                  openai_api_version=openai.api_version,
                                  openai_api_key=openai.api_key,
                                  openai_api_type=openai.api_type,
                                  deployment_name=self.deployment_name,
                                  temperature=0.7)
            self.chain = load_qa_chain(AzureOpenAI(
                deployment_name=self.deployment_name, model_name=self.model),
                                       chain_type="stuff")
        else:
            self.change_api_base('openai')
            llm = ChatOpenAI(temperature=0.7, openai_api_key=openai.api_key)
            self.chain = load_qa_chain(OpenAI(model_name=GPT_MODEL_NAME),
                                       chain_type="stuff")

        self.memory = ConversationSummaryBufferMemory(llm=llm,
                                                      max_token_limit=500)

        pinecone.init(api_key=os.environ["MY_PINECONE_API_KEY"],
                      environment=os.environ["MY_PINECONE_ENV"])

        self.index = Pinecone.from_existing_index(
            index_name='langchain-demo',
            embedding=OpenAIEmbeddings(model="text-embedding-ada-002"))

    def add_user_message_to_dialogue(self, message):
        self.dialogue.append({'role': 'user', 'content': message})

    def get_similiar_docs(self, query, k=2, score=False):
        index = self.index
        if score:
            similar_docs = index.similarity_search_with_score(query, k=k)
        else:
            similar_docs = index.similarity_search(query, k=k)
        return similar_docs

    def get_answer(self, query):
        similar_docs = self.get_similiar_docs(query)
        while True:
            try:
                answer = self.chain.run(input_documents=similar_docs,
                                        question=query)
                break
            except Exception as e:
                print('similiar_docs:', similar_docs)
                print('query:', query)
                print(e)
                self.update_openai_api_key()
        return answer

    def change_api_base(self, to_type):
        openai.api_type = to_type
        if to_type == 'azure':
            openai.api_version = os.environ["OPENAI_API_VERSION"]
            openai.api_base = os.environ["OPENAI_API_BASE"]
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            assert openai.api_type == "openai"
            openai.api_version = ""
            openai.api_base = 'https://api.openai.com/v1'
            self.update_openai_api_key()

    @staticmethod
    def load_openai_keys():
        with open(OPENAI_KEYS_FILE, "r") as f:
            context = f.read()
        return context.split('\n')

    def _load_state_prompt(self):
        state_prompt = self.prompt_handler.state_prompt()
        self.add_user_message_to_dialogue(state_prompt)
        return state_prompt

    def _load_task_prompt(self):
        task_prompt = self.prompt_handler.task_prompt()
        self.add_user_message_to_dialogue(task_prompt)
        return task_prompt

    def load_saved_dialogue(self, load_path=SAVED_DIALOGUE_FILE):
        # print("reading task prompt from {}".format(task_prompt_file))
        with open(load_path, "r") as f:
            self.dialogue = eval(f.read())

    def save_dialogue_to_file(self, save_path=SAVED_DIALOGUE_FILE):
        with open(save_path, "w", encoding='utf-8') as f:
            for message in self.dialogue:
                f.write(str(message) + '\n')

    def update_openai_api_key(self):
        if openai.api_type != 'openai':
            return

        curr_key = self.openai_api_keys[0]
        openai.api_key = os.environ["OPENAI_API_KEY"] = curr_key
        self.openai_api_keys.pop(0)
        self.openai_api_keys.append(curr_key)

    def check_if_the_taken_actions_list_needed_update(self,
                                                      check_content,
                                                      check_num=3,
                                                      top_k_charactors=0):
        if top_k_charactors == 0:
            if len(self.taken_actions_list) >= check_num:
                for i in range(check_num):
                    if self.taken_actions_list[-1 - i] == check_content:
                        if i == check_num - 1:
                            return True
                        else:
                            continue
                    else:
                        return False

            return False
        else:
            if len(self.taken_actions_list) >= check_num:
                for i in range(check_num):
                    if self.taken_actions_list[
                            -1 - i][:top_k_charactors] == check_content:
                        if i == check_num - 1:
                            return True
                        else:
                            continue
                    else:
                        return False

            return False

    def process_command(self, command_json, obs_input_prompt,
                        current_unit_name, current_avail_actions):
        '''
        manualAndHistorySearch
        askCurrentGameInformation
        finalDecision
        '''
        fc_logger.debug(f'Processing command: {command_json}')
        try:
            command_input = command_json['command']['input']
            command_name = command_json['command']['name']
        except Exception as e:
            print(e)
            print('Not in given json format, retrying...')
            if random.random() > 0.5:
                self.update_dialogue(obs_input_prompt +
                                     self.prompt_handler.insist_json(),
                                     pop_num=2)
                return None
            else:
                self.update_dialogue(obs_input_prompt, pop_num=2)
                return None
        if (command_name == 'finalDecision') and command_input['action']:
            # Here to implement controller
            print(command_input)
            exec_action = command_input['action']

            if command_input['action'] not in current_avail_actions:
                print('Not in the available action list, retrying...')
                # Here is the most time taking place.
                if random.random() > 0.5:
                    self.update_dialogue(
                        obs_input_prompt +
                        self.prompt_handler.insist_avail_action(),
                        pop_num=2)
                    return None
                else:
                    self.update_dialogue(obs_input_prompt, pop_num=2)
                    return None

            else:
                self.taken_actions_list.append(command_input['action'])
                if self.check_if_the_taken_actions_list_needed_update(
                        'goto', 15, 4
                ) or self.check_if_the_taken_actions_list_needed_update(
                        'keep_activity', 15, 0):
                    self.update_dialogue(
                        obs_input_prompt +
                        self.prompt_handler.insist_various_actions(
                            action="goto"),
                        pop_num=2)
                    # command_input = new_response['command']['input']
                    self.taken_actions_list = []
                    return None
                else:
                    print('exec_action:', exec_action)
                    return exec_action

        elif command_name == 'askCurrentGameInformation' and command_input[
                'query']:
            print(command_input)
            self.taken_actions_list.append('askCurrentGameInformation')
            return None

        elif command_name == 'manualAndHistorySearch' and command_input[
                'look_up']:
            print(command_input)

            if self.check_if_the_taken_actions_list_needed_update(
                    'look_up', 3, 0):
                answer = self.prompt_handler.generate("finish_look_for")
                print('answer:', answer)
                self.add_user_message_to_dialogue(answer)
                self.taken_actions_list = []
            else:
                query = command_input['look_up']
                answer = self.get_answer(query)
                print('answer:', answer)
                if random.random() > 0.5:
                    self.add_user_message_to_dialogue(
                        answer + self.prompt_handler.finish_look_for())
                else:
                    self.add_user_message_to_dialogue(answer)

                self.memory.save_context({'assistant': query},
                                         {'user': answer})
                self.taken_actions_list.append('look_up')

            return None
        else:
            print('error')
            print(command_json)

            if random.random() < 0.8:
                self.dialogue.pop(-1)
            else:
                self.add_user_message_to_dialogue(
                    'You should only use the given commands!')
            # self.update_dialogue(obs_input_prompt, pop_num = 1)

            return None

    def query(self, stop=None, temperature=0.7, top_p=0.95):
        self.restrict_dialogue()
        # TODO add retreat mech to cope with rate limit
        self.update_openai_api_key()

        fc_logger.debug(f'Querying with dialogue: {self.dialogue}')

        if self.model in ['gpt-3.5-turbo-0301', 'gpt-3.5-turbo']:
            assert openai.api_type == 'openai'
            response = openai.ChatCompletion.create(model=self.model,
                                                    messages=self.dialogue,
                                                    temperature=temperature,
                                                    top_p=top_p)
        elif self.model in ["gpt-35-turbo", "gpt-35-turbo-16k"]:
            assert openai.api_type == 'azure'
            response = openai.ChatCompletion.create(
                deployment_id=self.deployment_name,
                model=self.model,
                messages=self.dialogue)

        elif self.model in ['vicuna-33B']:
            local_config = {
                'temperature': temperature,
                'top_p': top_p,
                'repetition_penalty': 1.1
            }
            response = send_message_to_vicuna(self.dialogue, local_config)

        elif self.model in ['Llama2-70B-chat']:
            local_config = {
                'temperature': temperature,
                'top_p': top_p,
                'repetition_penalty': 1.1
            }
            response = send_message_to_llama(self.dialogue, local_config)

        else:
            response = openai.Completion.create(model=self.model,
                                                prompt=str(self.dialogue),
                                                max_tokens=1024,
                                                stop=stop,
                                                temperature=temperature,
                                                n=1,
                                                top_p=top_p)

        return response

    def update_dialogue(self, chat_content, pop_num=0):
        if pop_num != 0:
            for i in range(pop_num):
                self.dialogue.pop(-1)

        return self.communicate(chat_content)

    def parse_response(self, response):
        fc_logger.debug(f'Parsing response: {response}')

        if self.model in [
                'gpt-3.5-turbo-0301', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-0314'
        ]:
            return dict(response["choices"][0]["message"])

        elif self.model in ["gpt-35-turbo", "gpt-35-turbo-16k"]:
            try:
                ans = extract_json(
                    response['choices'][0]['message']['content'])
            except:
                return response["choices"][0]["message"]
            return {'role': 'assistant', 'content': ans}

        elif self.model in ['vicuna-33B', 'Llama2-70B-chat']:
            return {'role': 'assistant', 'content': extract_json(response)}

        else:
            # self.model in ['text-davinci-003', 'code-davinci-002']
            return {
                'role': 'assistant',
                'content': response["choices"][0]["text"][2:]
            }

    def restrict_dialogue(self):
        limit = TOKEN_LIMIT_TABLE[self.model]
        """
        The limit on token length for gpt-3.5-turbo-0301 is 4096.
        If token length exceeds the limit, we will remove the oldest messages.
        """
        # TODO validate that the messages removed are obs and actions
        while num_tokens_from_messages(self.dialogue, self.model) >= limit:
            temp_message = {}
            user_tag = 0
            if self.dialogue[-1]['role'] == 'user':
                temp_message = self.dialogue[-1]
                user_tag = 1

            while len(self.dialogue) >= 3:
                self.dialogue.pop(-1)

            while True:
                try:
                    self.add_user_message_to_dialogue(
                        'The former chat history can be summarized as: \n' +
                        self.memory.load_memory_variables({})['history'])
                    break
                except Exception as e:
                    print(e)
                    self.update_openai_api_key()

            if user_tag == 1:
                self.dialogue.append(temp_message)
                user_tag = 0

    def communicate(self, content, parse_choice_tag=False):
        self.add_user_message_to_dialogue(content)
        while True:
            try:
                raw_response = self.query()
                self.message = self.parse_response(raw_response)
                self.dialogue.append(self.message)

                response = self.message["content"]
                try:
                    response = json.loads(response)
                except Exception as e:
                    # self.dialogue.pop(-1)
                    print(e)
                    self.add_user_message_to_dialogue(
                        'You should only respond in JSON format as described')
                    print('Not response json, retrying...')

                    continue
                break

            except Exception as e:
                fc_logger.debug('Error in communicate: ' + str(e))
                fc_logger.debug('content: ' + content)
                print(e)
                print("retrying...")
                continue
        return response

    def reset(self):

        self.dialogue = []
        self.message = ''
        self.taken_actions_list = []

        self.openai_api_keys = self.load_openai_keys()
        self.state_prompt = self._load_state_prompt()
        self.task_prompt = self._load_task_prompt()
