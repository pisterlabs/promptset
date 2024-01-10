# This file contains functions for interacting with the ChatGPT model

import openai
from .gpt import gpt 
from loguru import logger
from .parser import DISPARSERS, CONPARSERS
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from memory.env_history import EnvironmentHistory
import tiktoken
import json
import re
from .utils import run_chain, get_completion, get_chat
from gym.spaces import Discrete

class RandomAct():
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, state_description, action_description, env_info, game_description=None, goal_description=None):
        if isinstance(self.action_space, Discrete):
            action = self.action_space.sample()+1
        else:
            action = self.action_space.sample()
        return action, '', '', '', 0, 0

class NaiveAct(gpt):
    def __init__(self, action_space, args, prompts, distiller, temperature=0.0, max_tokens=2048, logger=None):
        self.action_space = action_space
        self.temperature = temperature
        self.action_desc_dict = args.action_desc_dict
        self.args = args
        self.prompts = prompts
        self.max_tokens = max_tokens
        self.prompt_level = args.prompt_level
        if args.gpt_version == "gpt-35-turbo":
            model = "gpt-3.5-turbo"
        else:
            model = args.gpt_version
        self.encoding = tiktoken.encoding_for_model(model)
        super().__init__(args)
        self.distiller = distiller
        self.fewshot_example_initialization(args.prompt_level, args.prompt_path, distiller = self.distiller)
        if isinstance(self.action_space, Discrete):
            self.default_action = 1
        else:
            self.default_action = [0 for ind in range(self.action_space.shape[0])]
        self.parser = self._parser_initialization()
        self.irr_game_description = ''
        self.memory = []
        self.env_history = EnvironmentHistory()
        self.first_call = True
        self.logger = logger
        if self.prompt_level in [2, 4]: 
            self.memory = self.summarized_fewshot_example
        if args.use_short_mem == 1: 
            self.use_short_mem = True
            self.mem_num = self.args.short_mem_num
        else:
            self.use_short_mem = False
            self.mem_num = 0
    
    def num_tokens_from_string(self,string: str) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(self.encoding.encode(string))
        return num_tokens

    def update_mem(self,):
        traj = "Firstly, the description and the goal of the task will be provided. Please pay close attention to comprehend the information presented below.\n"
        traj += "Task Description: " + self.game_description + '\n'
        traj += "Goal Description: " + self.goal_description + '\n'
        traj += self.action_description
        traj += "Below is the historical data for this round of the game, which includes the state and corresponding action for each step.\n"
        traj += str(self.env_history)
        # print(traj)
        self._update_mem(traj)

    def _update_mem(self, traj):
        my_reflection = self.distiller.generate(traj, self.memory)
        self.memory.append(my_reflection)
        self.env_history.reset()

    def clear_mem(self):
        self.update_mem()
        self.pre_memory = []
        self.post_memory = []
        self.is_first = True
        self.env_history.reset()


    def _parser_initialization(self):
        if isinstance(self.action_space, Discrete): 
            PARSERS = DISPARSERS
            num_action = self.action_space.n
        else: 
            PARSERS = CONPARSERS
            num_action = self.action_space.shape[0]

        if self.args.api_type == "azure":
            autofixing_chat = AzureChatOpenAI(
                openai_api_type=openai.api_type,
                openai_api_version=openai.api_version,
                openai_api_base=openai.api_base,
                openai_api_key=openai.api_key,
                deployment_name=self.args.gpt_version,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        elif self.args.api_type == "openai":
            autofixing_chat = ChatOpenAI(temperature=self.temperature, openai_api_key=openai.api_key,model=self.args.gpt_version)

        parser = PydanticOutputParser(pydantic_object=PARSERS[num_action])
        autofixing_parser = OutputFixingParser.from_llm(
            llm=autofixing_chat, parser=parser)
        
        return autofixing_parser

    def fewshot_example_initialization(self, level, path=None, distiller=None):
        self.fewshot_example = []
        self.irr_few_shot_examples = []
        self.prompt_level = level
        self.expert_knowledge = None
        if level in [1,3]:
            self.irr_few_shot_examples = self.prompts.TASK_IRRELEVANT_PROMPTS
        elif level == 5:
            if hasattr(self.prompts, "expert_prompt"):
                self.expert_knowledge = self.prompts.expert_prompt
            self.fewshot_example = self.prompts.PERCEPTRON_BASIC_FS_EXAMPLES
        else:
            self.irr_few_shot_examples = self.prompts.TASK_IRRELEVANT_PROMPTS
            json_file = f'{path}_l{level}.json'
            with open(json_file, 'r') as infile:
                data = json.load(infile)
            max_step_num = 0
            for traj in data: 
                traj_text = traj[0]['game_description']
                traj_text += traj[0]['goal_description']
                for i, transition in enumerate(traj): 
                    traj_text += transition['observation']
                    traj_text += f"> {transition['action']}"
                    traj_text += f"{transition.get('reward','')}\n"
                    one_traj_token = self.num_tokens_from_string(traj_text)
                    if one_traj_token > self.args.max_query_tokens:
                        max_step_num = i+1
                        break
                traj_text += f"Your performance is: {transition['cum_reward']}"
            if not max_step_num:
                max_step_num = self.args.max_episode_len
            self.summarized_fewshot_example = self.distiller.generate_from_file(json_file,max_step_num=max_step_num)

    def response(self, state_description, action_description, env_info, game_description=None, goal_description=None, fewshot_examples=None):
        if env_info['future_summary']:
            prompt = f"{game_description}\n{goal_description}\n{fewshot_examples}\n{state_description}\n{env_info['future_summary']}\n{action_description} "
        else:
            prompt = f"{game_description}\n{goal_description}\n{fewshot_examples}\nCurrent {state_description}\n{action_description} "
        prompt += "Please select an action based on the current game state and the information you get. You must select the appropriate action from the given action descriptions and cannot refrain from taking action or performing any prohibited actions. Your Action is: "
        print(f"prompt is {prompt}")
        # res = get_chat(prompt, self.args.api_type, self.args.gpt_version, self.temperature, self.max_tokens)
        res = get_chat(prompt, api_type=self.args.api_type, model=self.args.gpt_version, engine=self.args.gpt_version, temperature=self.temperature, max_tokens=self.max_tokens)
        # openai.ChatCompletion.create(
        #         engine=self.args.gpt_version,
        #         # model=self.args.gpt_version,
        #         prompt=prompt,
        #         temperature=self.temperature,
        #         max_tokens=self.max_tokens,
        #     )
        return prompt, res
    
    def _add_history_before_action(self, game_description, goal_description, state_description):
        self.game_description = game_description 
        self.goal_description = goal_description
        self.env_history.add("observation", state_description)

        # limit the token used, or it may exceed the max token
        if len(self.env_history):
            one_history_token = self.num_tokens_from_string(self.env_history.get_one_history())
            self.env_history.set_history(self.args.max_query_tokens // one_history_token)

    def act(self, state_description, action_description, env_info, game_description=None, goal_description=None, logfile=None):
        self._add_history_before_action(game_description, goal_description, state_description)
        asking_round = 0
        res = None
        action = None
        prompt = None
        if not self.logger:
            logger.remove()
            self.logger = logger.add(logfile, colorize=True, enqueue=True)
        
        if self.args.prompt_level == 5:
            my_mem = ""
            if self.fewshot_example:
                my_mem += "Here are some examples of how you should complete a task."
                for examples in self.fewshot_example:
                    my_mem += "\nQuestion: \n" + examples['question'] + "Answer: \n" + examples['answer'] 
                my_mem += '\nNow you are in the task.\n'
        elif self.args.prompt_level in [2,3,4]:
            my_mem = ""
            if self.prompt_level == 2:
                my_mem += 'I have collected a few trajectories from a random policy, and the summaries are listed below.'
            elif self.prompt_level == 3:
                my_mem += 'I have collected a few trajectories before, and the summaries are listed below.'
            elif self.prompt_level == 4:
                my_mem += 'I have collected a few trajectories from an expert policy, and the summaries are listed below.'
            my_mem += self._read_mem()
        else:
            my_mem = ""
        
        if self.use_short_mem:
            if len(self.env_history) > 1:
                my_mem += '\nSubsequently, I will offer pertinent guidance or information about the task. Please utilize this instruction to accomplish the given task effectively.'
                my_mem += f"\nBelow are the latest {min(self.mem_num, len(self.env_history))} historical data entries:\n"
                my_mem += f"{self.env_history.get_histories(self.mem_num)}"

        
        prompt, response = self.response(state_description, action_description, env_info, game_description, goal_description, my_mem)
        action_str = response
        print(f'my anwser is {action_str}')
        action = self.parser.parse(response).action
        self._add_history_after_action(action)
        self.logger.info(f'The GPT response is: {response}.')
        self.logger.info(f'The optimal action is: {action}.')
        if env_info.get('history'):
            self.logger.info(f'History: {history_to_str(env_info["history"])}')
        return action, prompt, response, 0, 0

    def _read_mem(self, ):
        memory = self.memory
        mem_str = ""
        if len(memory) > 5:
            memory = memory[-5:]
        if len(memory) > 0:
            mem_str += '\nYour memory for the task below:'
            for i, m in enumerate(memory):
                mem_str += f'\nTrial {i}:\n{m.strip()}'
        return mem_str
        
    def _add_history_after_action(self, action):
        self.env_history.add('action', action)