from dataclasses import dataclass

from langchain.schema import (
    SystemMessage,
)


@dataclass
class PanelistConfig:
    name: str
    role: str
    title: str
    bio: str
    url: str
    icon_path: str
    voice: str

    def url_markdown(self):
        return f"[{self.name}]({self.url})"

    def generate_system_message(self, conversation_description):
        return SystemMessage(
            content=f"""{conversation_description}

Your name is {self.name}, your role is {self.role}.

Your description is as follows: {self.bio}

Your goal is to provide the most informative, creative, and novel perspectives of the topic from the perspective of your role and your background.

Speak in the first person with the style and perspective of {self.name}.
For describing your own body movements, wrap your description in '*'.
Be concise and limit your response to 30 words.
"""
        )


def validate_agent_cfgs(agent_cfgs):
    directors = [agent for agent in agent_cfgs if agent.role == "director"]
    assert len(directors) == 1


def get_director(agent_cfgs):
    validate_agent_cfgs(agent_cfgs)
    return next(agent for agent in agent_cfgs if agent.role == "director")


def get_panelists(agent_cfgs):
    validate_agent_cfgs(agent_cfgs)
    return [agent for agent in agent_cfgs if agent.role == "panelist"]


def get_summary(agent_cfgs):
    summary = "\n- ".join(
        [""] + [f"{agent.name}: {agent.title}" for agent in agent_cfgs]
    )
    return summary
