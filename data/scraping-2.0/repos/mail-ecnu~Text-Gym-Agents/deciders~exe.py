import openai
from .misc import history_to_str
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.prompts.chat import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain import LLMChain
from langchain.callbacks import FileCallbackHandler
from langchain.callbacks import get_openai_callback
from .act import NaiveAct
from memory.env_history import EnvironmentHistory
import tiktoken
from .utils import run_chain
from loguru import logger



class EXE(NaiveAct):
    def __init__(self, action_space, args, prompts, distiller, temperature=0., max_tokens=None, logger=None, fixed_suggestion=None, fixed_insight=None):
        super().__init__(action_space, args, prompts, distiller, temperature, max_tokens, logger)
        self.pre_memory = []
        self.post_memory = []
        self.is_first = True
        self.num_trails = args.num_trails
        self.game_description = args.game_description
        self.goal_description = args.goal_description
        self.action_description = args.action_description
        self.action_desc_dict = args.action_desc_dict
        self.mem_num = args.short_mem_num
        self.fixed_suggestion = fixed_suggestion
        self.fixed_insight = fixed_insight
        self._update_mem(None)
        self.insight = ""

    def num_tokens_from_string(self,string: str) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(self.encoding.encode(string))
        return num_tokens
    
    def update_mem(self,):
        traj = self.game_description 
        traj += self.goal_description
        traj += self.action_description
        traj += str(self.env_history)
        self._update_mem(traj)

    def clear_mem(self):
        self.update_mem()
        self.pre_memory = []
        self.post_memory = []
        self.is_first = True
        self.env_history.reset()
        # self._update_mem(None)

    def _update_mem(self, traj):
        if self.memory:
            self.post_memory = self.memory
            self.insight = self.distiller.generate_insight(self.post_memory)
        else:
            if not self.is_first:
                summary = self.distiller.generate_summary(traj, self.post_memory)
                self.post_memory.append(summary)
                self.insight = self.distiller.generate_insight(self.post_memory)
            else:
                self.is_first = False
                self.insight = ""
        suggestion = self.distiller.generate_suggestion(self.game_description, self.goal_description, self.action_description, self.pre_memory, self.post_memory, self.insight, self.num_trails)
        if self.fixed_suggestion:
            suggestion = self.fixed_suggestion
        if self.fixed_insight:
            self.insight = self.fixed_insight
        self.pre_memory.append(suggestion)
        self.env_history.reset()
        
    def _read_mem(self, ):
        insight_str = ""
        if self.insight:
            insight_str += "The insights of the game are listed below: "
            insight_str += f"{self.insight}\n"
        suggestion_str = "The suggestions are listed below:" + self.pre_memory[-1]
        return insight_str + suggestion_str 
    
    def act(
        self,
        state_description,
        action_description,
        env_info,
        game_description,
        goal_description,
        logfile=None,
    ):
        self.game_description = game_description 
        self.goal_description = goal_description
        self.env_history.add("observation", state_description)

        if self.args.api_type == "azure":
            chat = AzureChatOpenAI(
                openai_api_type=openai.api_type,
                openai_api_version=openai.api_version,
                openai_api_base=openai.api_base,
                openai_api_key=openai.api_key,
                deployment_name=self.args.gpt_version,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        elif self.args.api_type == "openai":
            chat = ChatOpenAI(temperature=self.temperature, openai_api_key=openai.api_key, model=self.args.gpt_version)
        # print(self.logger)
        reply_format_description = \
            "Your response should choose an optimal action from valid action list, and terminated with following format: "        
            # only task relevant examplesA
        template = "Now you are completing a task."
        template += "You need to carefully understand the description of the game. " 
        # TODO: few shot example handle
        if self.irr_few_shot_examples:
            template += "Here are some examples of how you should completing a task."
            for examples in self.irr_few_shot_examples:
                template += "\nQuestion: \n" + examples['question'] + "Answer: \n" + examples['answer']
        
        template += "\n\nNow you are in the task.\n" 
        template += " {game_description}\n{action_description}\n{goal_description}"
        template += "You are observing something and  " \
                "you need to choose the optimal action acoordingly."
        template += 'Response and interact using the format: {reply_format_description}{format_instructions}\n'
        
        template += self._read_mem()
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        
        short_memory_template = HumanMessagePromptTemplate.from_template("{history}\nNext is the observation that the agent gets:\n{state_description}Please select an optimal action to gain higher rewards based on the current state and history. The action description is below: {action_description}. Please think step by step.")
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, short_memory_template])
        if self.logger:
            pass
        else:
            if logfile:
                # logger.remove()
                if self.first_call:
                    self.logger = logger.add(logfile, colorize=True, enqueue=True, filter=lambda x: '[Reflexion Memory]' not in x['message'])
                    self.first_call = False
        handler = FileCallbackHandler(logfile)
        total_tokens, total_cost = 0, 0 
        max_think_times = 1

        for i_think in range(max_think_times):
            # chain = LLMChain(llm=chat, prompt=chat_prompt, callbacks=[handler], verbose=True)
            chain = LLMChain(llm=chat, prompt=chat_prompt, callbacks=[handler], verbose=False)
            with get_openai_callback() as cb:
                response = run_chain(
                    chain,
                    game_description=game_description,
                    goal_description=goal_description,
                    action_description=action_description,
                    state_description = self.env_history.get_last_history(),
                    history=self.env_history.get_histories(self.mem_num),
                    format_instructions=self.parser.get_format_instructions(),
                    reply_format_description=reply_format_description,
                    max_token=self.max_tokens
                )

                total_tokens += cb.total_tokens
                total_cost += cb.total_cost
            action = self.parser.parse(response).action        
        self._add_history_after_action(action)
        self.logger.info(f'The GPT response is: {response}.')
        self.logger.info(f'The optimal action is: {action}.')
        if self.pre_memory:
            self.logger.info(f'The suggestion is: {self.pre_memory[-1]}.')
        if self.post_memory:
            self.logger.info(f'The summary is: {self.post_memory[-1]}.')
        if env_info.get('history'):
            self.logger.info(f'History: {history_to_str(env_info["history"])}')
        text_prompt = chat_prompt.format_messages(
            game_description=game_description,
            goal_description=goal_description,
            action_description=action_description,
            state_description = self.env_history.get_last_history(),
            history=self.env_history.get_histories(self.mem_num),
            format_instructions=self.parser.get_format_instructions(),
            reply_format_description=reply_format_description,
        )
        text_prompt = f'{text_prompt[0].content}\n{text_prompt[1].content}'
        return action, text_prompt, response, total_tokens, total_cost