from agent.agents import Agent
from openai import OpenAI
from typing import Dict, List
from openai.types.beta.assistant_create_params import Tool
import json


class Manager(Agent):
    with open("agent/agent_config.json", "r") as file:
        agent_config = json.load(file)

    with open("team/member_config.json", "r") as file:
        member_config = json.load(file)

    def __init__(
        self,
        client: OpenAI,
        team_name: str,
        model: str = None,
        file_names: List[str] = None,
    ):
        super().__init__(
            client=client,
            model=model,
            name=team_name,
            instructions=Manager.agent_config["manager"]["instructions"],
            tools=Manager.agent_config["manager"]["tools"],
            file_names=file_names,
        )
        self.members: Dict[str, Agent] = {}
        self.member_names = Manager.member_config[self.name]

        for agent_name in self.member_names:
            self.members[agent_name] = Agent.get_or_create(
                client=self.client, **Manager.agent_config[agent_name]
            )

    @classmethod
    def get_or_create(
        cls,
        client: OpenAI,
        team_name: str,
        model: str = None,
        file_names: List[str] = None,
    ):
        if team_name not in Agent._instances:
            # If the instance doesn't exist, create and store it
            instance = cls(client, team_name, model, file_names)
            Agent._instances[team_name] = instance
        else:
            print(f"Returning existing instance of Agent {team_name}")

        return Agent._instances[team_name]

    def upload_team_config(self):
        team_config = {
            k: Manager.agent_config[k]
            for k in self.member_names
            if k in Manager.agent_config
        }

        with open(f"files/{self.name}.json", "w") as file:
            json.dump(team_config, file)

        self.add_files([f"{self.name}.json"])

    assign_task_json = {
        "name": "assign_task",
        "description": "Call this function to assign a specific task to the team member that you think is most suitable for the task. The function will return the response from the team member.",
        "parameters": {
            "type": "object",
            "properties": {
                "team_member": {
                    "type": "string",
                    "description": "The team member that you think is most suitable for solving the task.",
                },
                "task": {
                    "type": "string",
                    "description": "The task to be assigned. This should be a clear request that the team member can handle.",
                },
            },
            "required": ["team_member", "task"],
        },
    }

    def assign_task(self, team_member: str, task: str):
        res = self.members[team_member].process_message(task)
        return res
