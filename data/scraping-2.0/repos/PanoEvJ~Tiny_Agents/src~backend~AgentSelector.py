import os
from dotenv import load_dotenv
load_dotenv()

import openai
from typing import List
from pydantic import BaseModel

# # write an openai function that takes ensures response is a list
# class Response(BaseModel):
#     response: List[str]

# tools = [
#   {
#     "type": "function",
#     "function": {
#       "name": "list_agents",
#       "description": "ensure response is a list",
#       "parameters": Response.model_json_schema(),
#     }
#   }
# ]


class AgentSelector:
    def __init__(
        self,
        task: str,
        available_agents: List,
        n_agents: int = 3,
        chat_history: str = '',
        prev_output: str = '',
    ):
        self.task = task
        self.available_agents = available_agents
        self.n_agents = n_agents
        self.chat_history = chat_history
        self.prev_output = prev_output
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.agents_to_use = []
        self.inputs = {
            "task": task,
            "available_agents": available_agents,
            "n_agents": n_agents,
            "chat_history": chat_history,
            "prev_output": prev_output,
        }

    def select_agents(self):
        # Use OpenAI GPT-4 API to select agents
        openai.api_key = self.api_key
        client = openai.OpenAI()
        system_prompt=f"You are an Agent Selector. Select {self.n_agents} agents from the available list of agents to best solve the task based on their descriptions. ONLY RESPOND WITH A PYTHON LIST OF AGENTS."
        # print(self.available_agents)
        user_prompt=f"""
                    Task: {self.task}\n
                    Available Agents: {self.available_agents}\n
                    Number of Agents to Select: {self.n_agents}\n\n
                    List: 
                """
        
        # tools: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
        # tools = []

        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Task: Please help me with my Digital Marketing.\nAvailable Agents: {self.available_agents}\nNumber of Agents to Select: {self.n_agents}\n\nList: "},
                {"role": "assistant", "content": "marketing_digital"},
                {"role": "user", "content": user_prompt}
            ],
            # max_tokens=100,
            # tools=tools,
            # tool_choice="auto",
            # stream=True,
        )

        # if stream is True, then response.choices will be a generator
        # for chunk in response:
        #     print(chunk.choices[0].delta)

        self.agents_to_use = response.choices[0].message.content
        return self.agents_to_use

    def run_selection(self):
        selected_agents = self.select_agents()
        selected_agents = selected_agents.replace("[", "").replace("]", "").replace("'", "").replace(" ", "").split(",")
        print(f"Agents Selected: {selected_agents}")
        print(f"return type: {type(selected_agents)}")
        return selected_agents

if __name__ == "__main__":
    example = "exampleJuan"
    if example == "example1":
        task = "Please help me with my sales."
        available_agents = ["marketing_seo", "marketing_digital", "marketing_miami", "sales_chad", "sales_brad", "sales_senior"]
        n_agents=1
    elif example == "exampleJuan":
        task = "revenue in the east coast is falling and the competitor is doing great with their product."
        available_agents = ["marketing", "sales", "finance", "engineering", "customer_service", "hr", "legal", "operations", "product"]
        n_agents=3

    
    print(f"Task: {task}")
    # agent_selector = AgentSelector(task: str, available_agents: List[str], n_agents: int=n_agents)
    agent_selector = AgentSelector(task, available_agents, n_agents=n_agents)
    agent_selector.run_selection()
    # print(agent_selector.inputs)
