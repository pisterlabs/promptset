import os
from typing import Optional

from src.configure import AgentBehaviorParameters
from src.llms import CachedChatOpenAI
from src.messages import Messages
from src.tot.openai_llms import OpenAILanguageModel
from src.tot.treeofthoughts import TreeofThoughtsBFS

num_thoughts = 2
max_steps = 2
max_states = 2
pruning_threshold = 0.5


class GptOr:
    def __init__(self, agent_params: AgentBehaviorParameters, conversations: Messages):
        self.agent_params = agent_params
        self.conversations = conversations
        self.llm: Optional[CachedChatOpenAI] = CachedChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4"
        )
        self.tot = TreeofThoughtsBFS(
            OpenAILanguageModel(api_key=os.getenv("OPENAI_API_KEY"))
        )

    def get_tot_response(self, messages):
        msg = ""
        for message in messages[-2:]:
            msg += message.content
        response = self.tot.solve(
            initial_prompt=msg,
            num_thoughts=num_thoughts,
            max_steps=max_steps,
            pruning_threshold=pruning_threshold,
            max_states=max_states,
            value_threshold=0.9,
        )
        return response[0]

    def generate_problem_formulation(self, use_tot=False):
        formulation_messages = self.conversations.get_formulation_conversation()
        if use_tot:
            llm_response = self.get_tot_response(formulation_messages)
        else:
            output = self.llm(messages=formulation_messages)
            llm_response = output.content

        self.conversations.global_conversations.append(llm_response)
        self.conversations.formulation_response = llm_response
        return llm_response

    def generate_problem_code(self, use_tot=False):
        code_generation_messages = self.conversations.get_code_conversation()
        if use_tot:
            llm_response = self.get_tot_response(code_generation_messages)
        else:
            output = self.llm(messages=code_generation_messages)
            llm_response = output.content

        self.conversations.global_conversations.append(llm_response)
        return llm_response

    def generate_codefix_formulation(self, execution_status, use_tot=True):
        codefix_conversations = self.conversations.get_code_fix_conversation(
            execution_status
        )
        if use_tot:
            llm_response = self.get_tot_response(codefix_conversations)
        else:
            output = self.llm(messages=codefix_conversations)
            llm_response = output.content
        self.conversations.global_conversations.append(llm_response)
        return llm_response

    def generate_test(self):
        pass

    @property
    def automatic_test(self):
        if (
            self.agent_params["Debug"]
            and not self.agent_params["Test"]
            and not self.agent_params["Human"]
        ):
            return True
        return False
