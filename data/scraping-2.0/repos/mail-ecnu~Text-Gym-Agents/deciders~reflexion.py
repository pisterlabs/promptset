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
from loguru import logger
from langchain.callbacks import FileCallbackHandler
from langchain.callbacks import get_openai_callback
from .act import NaiveAct
from memory.env_history import EnvironmentHistory
import tiktoken
from .utils import run_chain


class Reflexion(NaiveAct):
    def __init__(self, action_space, args, prompts, distiller, temperature=0.1, max_tokens=None, logger=None):
        super().__init__(action_space, args, prompts, distiller, temperature, max_tokens, logger)
    
    def num_tokens_from_string(self,string: str) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(self.encoding.encode(string))
        return num_tokens
    
    def update_mem(self,):
        traj = self.game_description 
        traj += self.goal_description
        one_history_token = self.num_tokens_from_string(self.env_history.get_one_history())
        history_num = self.args.max_query_tokens // one_history_token
        traj += self.env_history.get_histories_with_last(history_num)
        self._update_mem(traj)

    def _update_mem(self, traj):
        my_reflection = self.distiller.generate(traj, self.memory)
        self.memory.append(my_reflection)
        self.env_history.reset()

    def act(
        self,
        state_description,
        action_description,
        env_info,
        game_description,
        goal_description,
        logfile=None,
    ):
        self.action_description = action_description
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
        suffix_flag = False
        reply_format_description = \
            "Your response should choose an optimal action from a valid action list and terminate with the following format: "

        # System Message
        human_template = "Now, you are completing a challenging task. You must carefully understand the Reflexion method you will use and apply it to the following task.\n"
        
        # task-irrelevant SystemMessage
        if self.irr_few_shot_examples:
            human_template += 'In the following example, I shall present a set of question and answer about the Reflexion method. Please adhere to the format and reasoning of the provided response when addressing the subsequent task.\n'
            for i, examples in enumerate(self.irr_few_shot_examples):
                human_template += f"\nExample {i+1}:\n"
                human_template += "Question: \n" + examples['question'] + "\nAnswer: \n" + examples['answer']

        # task-irrelevant few shot if have
        if self.irr_few_shot_examples:
            human_template += "\nMoving forward, I will describe the task, the goal, and the actions you may execute. Please pay close attention to comprehend the information presented below.\n"

        if self.fewshot_example:
            human_template += "I will describe the task, the goal, and the actions you may execute. Please pay close attention to comprehend the information presented below."
        # print(fewshot_example_prompt.format(**fewshot_examples[0]))
        human_template += '\nTask Description: {game_description} \n'
        human_template += 'Goal Description: {goal_description}\n'
        human_template += 'Actions Description: {action_description}\n'

        if self.fewshot_example:
            human_template += "Here, I will provide you with some guidance to help you better understand the rules of the task. Next are some examples: "
            for i, examples in enumerate(self.fewshot_example):
                human_template += f"\nExample {i+1}:\n"
                human_template += "Question: \n" + examples['question'] + "\nAnswer: \n" + examples['answer']

        if self.prompt_level in [2, 3, 4]:
            if self.memory:
                human_template += '\nSubsequently, I will offer pertinent guidance or information about the task. Please utilize this instruction to accomplish the given task effectively.\n'
                suffix_flag = True
                if self.prompt_level == 2:
                    human_template += 'I have collected a few trajectories from a random policy, and the summaries are listed below.'
                elif self.prompt_level == 3:
                    human_template += 'I have collected a few trajectories before, and the summaries are listed below.'
                elif self.prompt_level == 4:
                    human_template += 'I have collected a few trajectories from an expert policy, and the summaries are listed below.'
                human_template += self._read_mem() + "\n"

        if self.use_short_mem:
            if len(self.env_history) > 1:
                if not suffix_flag: 
                    human_template += '\nSubsequently, I will offer pertinent guidance or information about the task. Please utilize this instruction to accomplish the given task effectively.'
                human_template += f"\nBelow are the latest {min(self.mem_num, len(self.env_history))} historical data entries:\n"
                human_template += f"{self.env_history.get_histories(self.mem_num)}"
        human_template += '\nNext is the observation that the agent gets:\nCurrent {state_description}\n'
        human_template += 'Please select an action based on the current game state and the information you get. You must select the appropriate action from the given action descriptions and cannot refrain from taking action or performing any prohibited actions. Here is the action description below:\n{action_description}\n'
        human_template += 'Also, please keep in mind not to answer with any redundant and irrelevant content.\n'
        human_template += "Finally, you also need to normalize your output according to the reply format description.\n"
        human_template += 'Reply format description: {reply_format_description}{format_instructions}\n'

        human_message_prompt = PromptTemplate(
            template=human_template,
            input_variables=[
                'state_description', 'goal_description', 'game_description',
                'action_description', 'reply_format_description'],
            partial_variables={'format_instructions': self.parser.get_format_instructions()}
        )

        human_message_prompt = HumanMessagePromptTemplate(prompt=human_message_prompt)
        
        chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
        if not self.logger:
            # logger.remove()
            if self.first_call:
                self.logger = logger.add(logfile, colorize=True, enqueue=True, filter=lambda x: '[Reflexion Memory]' not in x['message'])
                self.first_call = False
        handler = FileCallbackHandler(logfile)
        total_tokens, total_cost = 0, 0 
        max_think_times = 1
        # TODO: ADD REACT Support
        # print(str(self.env_history))

        for i_think in range(max_think_times):
            chain = LLMChain(llm=chat, prompt=chat_prompt, callbacks=[handler], verbose=False)
            with get_openai_callback() as cb:
                response = run_chain(
                    chain,
                    state_description=self.env_history.get_last_history(),
                    game_description=game_description,
                    goal_description=goal_description,
                    action_description=action_description,
                    format_instructions=self.parser.get_format_instructions(),
                    reply_format_description=reply_format_description,
                    max_token = self.max_tokens
                )

                total_tokens += cb.total_tokens
                total_cost += cb.total_cost
            action = self.parser.parse(response).action
        text_prompt = chat_prompt.format_messages(
            state_description=self.env_history.get_last_history(),
            game_description=game_description,
            goal_description=goal_description,
            action_description=action_description,
            format_instructions=self.parser.get_format_instructions(),
            reply_format_description=reply_format_description,
        )
        texts = ""
        for text in text_prompt:
            texts += text.content + "\n"

        self._add_history_after_action(action)
        self.logger.info(f'The GPT response is: {response}.')
        self.logger.info(f'The optimal action is: {action}.')
        if self.memory:
            self.logger.info(f'The memory is: {self.memory[-1]}.')
        if env_info.get('history'):
            self.logger.info(f'History: {history_to_str(env_info["history"])}')

        return action, texts, response, total_tokens, total_cost
