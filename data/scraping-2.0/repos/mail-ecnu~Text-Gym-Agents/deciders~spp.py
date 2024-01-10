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
from .utils import run_chain

class SPP(NaiveAct):
    def __init__(self, action_space, args, prompts, distiller, temperature=0.1, max_tokens=None, logger=None):
        super().__init__(action_space, args, prompts, distiller, temperature, max_tokens, logger)

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
        self._add_history_before_action(game_description, goal_description, state_description)

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

        self.fewshot_example = self.irr_few_shot_examples if not self.fewshot_example else self.fewshot_example
        self.irr_few_shot_examples = self.irr_few_shot_examples if not self.fewshot_example else self.fewshot_example
        suffix_flag = False
        reply_format_description = \
            "Your response should choose an optimal action from a valid action list and terminate with the following format: "

        # System Message
        human_template = "When faced with a task, begin by identifying the participants who will contribute to solving the task. Then, initiate a multi-round collaboration process until a final solution is reached. The participants will give critical comments and detailed suggestions whenever necessary.\n"
        human_template += "Now, you are completing a challenging task. You must carefully understand the Solo-Performance-Prompting method you will use and apply it to the following task.\n"
        
        # task-irrelevant SystemMessage
        if self.irr_few_shot_examples:
            human_template += 'In the following example, I shall present a set of question and answer with the Solo-Performance-Prompting method. Please adhere to the format and reasoning of the provided response when addressing the subsequent task.\n'
            for i, examples in enumerate(self.irr_few_shot_examples):
                human_template += f"\nExample {i+1}:\n"
                human_template += "Question: \n" + examples['question'] + "\nAnswer: \n" + examples['answer']

        # task-irrelevant few shot if have
        if self.irr_few_shot_examples:
            human_template += "\nMoving forward, I will describe the task, the goal, and the actions you may execute. Please pay close attention to comprehend the information presented below.\n"

        human_template += '\nTask Description: {game_description} \n'
        human_template += 'Goal Description: {goal_description}\n'
        human_template += 'Actions Description: {action_description}\n'

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
        human_template += 'Please note that you need to carefully lay out the participants who will contribute to solving the task and initiate a multi-round collaboration process until a final solution is reached. Now, identify the participants and collaboratively solve the following task step by step.Also, please keep in mind not to answer with any redundant and irrelevant content.\n'
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
            logger.remove()
            self.logger = logger.add(logfile, colorize=True, enqueue=True)
        handler = FileCallbackHandler(logfile)

        chain = LLMChain(llm=chat, prompt=chat_prompt, callbacks=[handler], verbose=False)

        with get_openai_callback() as cb:
            response = run_chain(
                chain,
                game_description=game_description,
                state_description=state_description,
                goal_description=goal_description,
                action_description=action_description,
                reply_format_description=reply_format_description
            )
            total_tokens = cb.total_tokens
            total_cost = cb.total_cost
        action = self.parser.parse(response).action

        text_prompt = chat_prompt.format_messages(
            game_description=game_description,
            state_description=state_description,
            goal_description=goal_description,
            action_description=action_description,
            reply_format_description=reply_format_description
        )
        texts = ""
        for text in text_prompt:
            texts += text.content + "\n"

        self._add_history_after_action(action)

        self.logger.info(f'The GPT response is: {response}.')
        self.logger.info(f'The optimal action is: {action}.')
        if env_info.get('history'):
            self.logger.info(f'History: {history_to_str(env_info["history"])}')

        return action, texts, response, total_tokens, total_cost
