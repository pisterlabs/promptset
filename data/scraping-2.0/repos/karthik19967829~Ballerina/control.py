import time
from abc import ABC, abstractmethod
import os 
import openai 

prompt_template = """
Given the following examples of task execution plan with agents Schedule_Agent and Search_Agent, generate the task execution plan for the given query:

Example 1:
Input: "find a concert nearby"
Task Execution Plan:
Call Search_Agent("find a concert nearby") 
The response of the Search_Agent is "Events available:Mexican concert at 2PM and Hindustani concert 5PM"
Call Schedule_Agent("Find available slots among the time slots among the user's schedule [2PM,5PM]")
The response from Schedule_Agent is 2 PM is available 
Call Book_Agent("Book Mexican concert at 2 PM")

Example 2:
Input: "find a concert nearby"
Task Execution Plan:
Call Search_Agent("find a science meetup nearby") 
The response of the Search_Agent is "Events available:LLM event at 1 PM on thursday and Neurosciene event at 3 PM Friday "
Call Schedule_Agent("Find available slots among the time slots among the user's schedule [1PM Thursday,3PM Friday]")
The response from Schedule_Agent is 1 PM Thursday is available 
Call Book_Agent("Book LLM event at 1 PM Thursday")

Example 3:
Based on the above examples, Generate the execution plan for the following:
Input: "{}"
"""


class Agent(ABC):

    def __init__(self, llm, context_len=2000):
        self.type = self.agent_type()  # Abstract property
        self.life_label = time.time()
        self.name = f"{self.type}_{self.life_label}"
        self.llm = llm
        self.context_len = context_len
        self.task = None

    @abstractmethod
    def agent_type(self):
        pass

    @abstractmethod
    def action_parser(self, text, available_actions):
        pass

    @abstractmethod
    def prompt_layer(self, control_prompt, available_actions):
        pass

    def llm_layer(self, prompt):
        return self.llm(prompt)

    def forward(self, control_prompt, available_actions):
        prompt = self.prompt_layer(control_prompt, available_actions)
        action = self.llm_layer(prompt).lstrip(' ')
        action = self.action_parser(action, available_actions)
        return action

class ControlAgent:
    def __init__(self, agents):
        self.type = "Control_Agent"
        self.agents = agents
        self.task = {}
        self.actions = {}
        self.observations = {}
        self.agent_calls = {} # saving the agents types in sessions
        self.item_recall = {}
        self.cur_session = 0
        self.agents = agents
        self.control_dict = control_dict
        
    def new_session(self, idx, task):
        self.cur_session = idx
        self.task[idx] = task
        self.actions[idx] = ['reset']
        self.observations[idx] = []
        self.item_recall[idx] = []
        self.agent_calls[idx] = ["Search_Click_Controller"]
        self.task_assign(task)
    
    def task_assign(self, task):
        for agent in self.agents:
            agent.task = task

    def action_parser(self, text, available_actions):
        nor_text = text.strip().lower()
        if nor_text.startswith('search'):
            query = get_query(nor_text)
            return f"search[{query}]"
        if nor_text.startswith('click'):
            if 'click' in available_actions:
                for a in available_actions['click']:
                    if a.lower() in nor_text:
                        return f"click[{a}]"
        return text
    
    def get_agents_types(self):
        return [a.type for a in self.agents]
    
    
    def call_agent(self, observation, available_actions):
        for agent in self.agents:
            if self.control_dict[agent.type] in available_actions:
                return agent
    
    def llm_layer(self, prompt):
        return self.llm(prompt)
    
    def forward(self, observation, available_actions=None):
        self.observations[self.cur_session].append(observation)
        agent = self.call_agent(observation, available_actions)
        prompt = self.prompt_layer(agent, available_actions)
        action = agent.forward(prompt, available_actions).lstrip()
        action = self.action_parser(action, available_actions)
        self.actions[self.cur_session].append(action)
        self.agent_calls[self.cur_session].append(agent.type)
        return action


if __name__=="__main__":
    openai.api_key = os.environ["OPENAI_API_KEY"]

    response = openai.ChatCompletion.create(  # type: ignore
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt_template.format("find events for Monday")}],
        temperature=0.1,
        max_tokens=200,
        top_p=0.95,
    )
    answer = response["choices"][0]["message"]["content"]
    print(answer)
