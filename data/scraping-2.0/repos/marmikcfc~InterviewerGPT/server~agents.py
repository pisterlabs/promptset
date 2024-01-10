from pydantic import Field
from langchain.chains import ConversationChain
from langchain.agents import Agent, AgentOutputParser
from collections import defaultdict
from interview_sections import InterviewSections

class InterviewAgent():
    """A conversational agent that uses the ConversationChain for interactions and maintains chat history."""
    def __init__(self):
        # Set the memory for the ConversationChain
        self.current_chain_id = InterviewSections.CODING_INTERVIEW_INTRO
        self.chain_id_dict = defaultdict(ConversationChain)
    
    def add_chain(self, chain_id, chain) -> bool:
        self.chain_id_dict[chain_id] = chain
        return True
    
    def get_current_chain(self) -> ConversationChain:
        return self.chain_id_dict[self.current_chain_id]

    def set_current_chain(self, chain_id: InterviewSections) -> bool:
        if chain_id in self.chain_id_dict:
            self.current_chain_id = chain_id
            return True
        return False
    
    def update_chains(self, chain_id: int, chain: ConversationChain) -> bool:
        if chain_id in self.chain_id_dict:
            self.chain_id_dict[chain_id] = chain
            return True
        return False

    def get_current_chain_id(self) -> int:
        return self.current_chain_id
    
    def get_current_interview_section(self):
        return self.current_chain_id
    

    def get_chain(self, chain_id: int) -> ConversationChain:
        if chain_id in self.chain_id_dict:
            return self.chain_id_dict[chain_id]


    def interact(self, message: str) -> str:
        """
        Interact with the agent using a given message.

        Args:
        - message (str): The input message for the agent.

        Returns:
        - str: The agent's response.
        """
        response = self.get_current_chain(message)
        return response