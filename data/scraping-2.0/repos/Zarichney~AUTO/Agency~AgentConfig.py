# /Agency/AgentConfig.py

import json
import os
import openai
from Agents import agent_classes
from Utilities.Config import agent_config_file_name
from Utilities.Log import Debug
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Agents.BaseAgent import BaseAgent

class AgentConfig:
    def __init__(self, agent_id, agent_name):
        self.agent_id: str = agent_id
        self.agent_name: str = agent_name

    def delete(self, client: openai):
        try:
            client.beta.assistants.delete(self.agent_id)
            Debug(f"OpenAI Assistant '{self.agent_name}' Deleted")
        except Exception:
            Debug(f"Failed to delete OpenAI Assistant '{self.agent_name}'")
            pass
        
    def to_dict(self):
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name
        }


class AgentConfigurationManager:
    def __init__(self, agency, rebuild_agents=False):
        self.agency = agency
        self.configurations: [AgentConfig] = []
        self.agents: [BaseAgent] = []

        # if config_file does not exist or has empty contents, initialize file
        if (
            not os.path.exists(agent_config_file_name)
            or os.stat(agent_config_file_name).st_size == 0
        ):
            self._write_to_config_file()

        self._load_from_session_file()

        if len(self.configurations) > 0 and rebuild_agents:
            self._reset_config_file()

        if len(self.configurations) == 0:
            self._setup_agents()
        
        self._load_agents()
        
    def _reset_config_file(self):
        for agentConfig in self.configurations:
            agentConfig.delete(self.agency.client)
        self.configurations = []
        self._write_to_config_file()

    def _write_to_config_file(self):
        with open(agent_config_file_name, "w") as config_file:
            configurations_dict = [config.to_dict() for config in self.configurations]
            config_file.write(json.dumps({"agents": configurations_dict}) + "\n")

    def _load_from_session_file(self):
        self.configurations = []
        with open(agent_config_file_name, "r") as config_file:
            config = json.load(config_file)
            for config_dict in config["agents"]:
                self.configurations.append(
                    AgentConfig(
                        agent_id=config_dict["agent_id"],
                        agent_name=config_dict["agent_name"],
                    )
                )

    def _load_agents(self):
        for agent_config in self.configurations:
            self.agents.append(
                self._initialize_agent(
                    agent_config.agent_name, agent_config.agent_id
                )
            )
            
        # Agent creation expected. Assert or return error if agent_id cannot be found
        for agent_config in self.configurations:
            if agent_config.agent_id not in [agent.id for agent in self.agents]:
                raise Exception(
                    f"Session Loading Error: Agent {agent_config.agent_name} with id {agent_config.agent_id} could not be added to agency."
                )

    def _initialize_agent(self, agent_name:str, agent_id:str=None):
        if agent_name not in agent_classes:
            raise Exception(f"Invalid agent type: {agent_name}")

        AgentClass = agent_classes[agent_name]
        agent_instance = AgentClass(self.agency, agent_id)
        return agent_instance

    def _setup_agents(self):

        # Generate new instances of agents
        for agent_name, AgentClass in agent_classes.items():
            agent_instance = self._initialize_agent(agent_name)
            self.agents.append(agent_instance)
            self.configurations.append(
                AgentConfig(agent_instance.id, agent_name)
            )
            
        self._write_to_config_file()
        Debug("Generated agents written to file")
