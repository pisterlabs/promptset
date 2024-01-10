# /Tools/Delegate.py

from instructor import OpenAISchema
from pydantic import Field
from Utilities.Log import Log, Debug, type
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Agents.BaseAgent import BaseAgent
    from Agency.Agency import Agency

class Delegate(OpenAISchema):
    """
    Used to appoint another agent as the agency's active agent.
    To hand off the responsibility of tackling the current action item to different specialized agent.
    """

    recipient_name: str = Field(
        ...,
        description="The agent's name which is being requested for assistance",
    )
    instruction: str = Field(
        ...,
        description="Specify the task required for the recipient agent to complete. Recall the agency's plan and speak to the assistant in terms of the action items you want them to complete.",
    )

    def run(self, agency: 'Agency'):
        
        recipient: 'BaseAgent' = agency.get_agent(self.recipient_name)
        current_agent: 'BaseAgent' = agency.active_agent
        
        if recipient.name == current_agent.name:
            Log(type.ERROR, f"{recipient.name} attempted to delegate to itself")
            return "You cannot delegate to yourself. Supply a different agent name instead."

        prompt = f"# User's Prompt\n"
        prompt += f"{agency.prompt}\n\n"
        
        # Every fifth delegation, the agency will remind the agent of the plan
        if agency.delegation_count % 5 == 0:
            prompt += f"# Agency's Plan\n"
            prompt += f"{agency.plan}\n\n"
        
        prompt += f"I, {current_agent.name}, am seeking assistance from you, Agent {recipient.name}.\n"
        prompt += "According to our agency's mission, could you perform the following please:\n"
        prompt += self.instruction

        Log(type.COMMUNICATION, f"{current_agent.name} is prompting {recipient.name}:\n{self.instruction}")
        Debug(f"{current_agent.name} is delegating to {recipient.name} with this prompt:\n{prompt}")

        agency.add_message(message=prompt)

        agency.active_agent = recipient
        agency.delegation_count += 1
        current_agent.task_delegated = True

        return "Delegation complete. The recipient will complete the task. Do not use any tools. Just respond that you've delegated"
