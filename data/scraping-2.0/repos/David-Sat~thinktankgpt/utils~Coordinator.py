
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from typing import List, Dict
from utils.Worker import Worker
from langchain.schema import StrOutputParser


example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )

class Coordinator(Worker):
    def __init__(self, model, num_experts, topic, stance):
        super().__init__(model=model)
        self.num_experts = num_experts
        self.topic = topic
        self.stance = stance
        self.system_prompts = self.config["coordinator"]['system_prompts']
        self.examples = self.config["coordinator"]['examples']


    def generate_expert_instructions(self) -> List[Dict]:
            list_prompt = self.create_list_prompt(self.num_experts, self.topic, example_prompt)
            chain1 = (
                list_prompt
                | self.model
                | StrOutputParser()
            )

            expert_list = self.process_expert_list(chain1.invoke({}))
            batch_inputs = []
            for expert in expert_list:
                batch_inputs.append({"role": expert["role"], "stance": expert["stance"], "topic": self.topic})
            
            expert_instruction_prompt = self.create_expert_instruction_prompt(example_prompt)
            chain2 = (
                expert_instruction_prompt
                | self.model
                | StrOutputParser()
            )

            expert_instructions = chain2.batch(batch_inputs)

            for i, expert in enumerate(expert_list):
                expert["instructions"] = expert_instructions[i]

            return expert_list

    def process_expert_list(self, expert_list_raw: str) -> List[Dict[str, str]]:
        expert_list = expert_list_raw.split("; ")
        result_list = []
        for expert in expert_list:
            role, avatar, stance = expert.split(", ")
            result_list.append({"role": role, "avatar": avatar, "stance": stance})
        return result_list


    def create_list_prompt(self, num_experts: int, topic: str, example_prompt: str) -> ChatPromptTemplate:
            coordinator_list_instruction = self.system_prompts['system1'].replace("##num_workers##", str(num_experts))
            few_shot_prompt_list_experts = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                examples=self.examples['examples1'],
            )
            human_query = f"{num_experts} experts for the topic: {topic}"
            list_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", coordinator_list_instruction),
                    few_shot_prompt_list_experts,
                    ("human", human_query),
                ]
            )
            return list_prompt


    def create_expert_instruction_prompt(self, example_prompt: str) -> ChatPromptTemplate:
            few_shot_prompt_instruct_expert = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                examples=self.examples['examples2'],
            )
            expert_instruction_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_prompts['system2']),
                    few_shot_prompt_instruct_expert,
                    ("human", "role: {role}; stance: {stance}; topic: {topic}"),
                ]
            )
            return expert_instruction_prompt
