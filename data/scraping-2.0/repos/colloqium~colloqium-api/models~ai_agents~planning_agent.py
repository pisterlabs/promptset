from models.ai_agents.agent import Agent
from models.interaction import SenderVoterRelationship
from context.database import db
from tools.ai_functions.create_texting_agent import CreateTextingAgent
from tools.ai_functions.create_robo_caller_agent import CreateRoboCallerAgent
from tools.ai_functions.create_email_agent import CreateEmailAgent
from tools.utility import get_llm_response_to_conversation, initialize_conversation
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
import json

name = "planning_agent"

class PlanningAgent(Agent):
    __mapper_args__ = {
        'polymorphic_identity': name
    }



    def __init__(self, sender_voter_relationship_id: int):
        
        print(f"Creating a new PlanningAgent for sender_voter_relationship_id {sender_voter_relationship_id}")
        
        sender_voter_relationship = SenderVoterRelationship.query.get(sender_voter_relationship_id)

        prompt_template = '''
            You are an agent for political campaign. Your job is to decide the best way to engage with voters to get them to vote for your candidate.

            You know the following about the candidate: {candidate_info}

            You know the following about the voter: {voter_info}

            If you make a mistake calling a function, try to call it again at least once. Respond "Ready" if you are ready to begin and wait for a request from the campaign manager.
        '''

        system_prompt_template = SystemMessagePromptTemplate.from_template(prompt_template)

        chat_prompt_template = ChatPromptTemplate.from_messages([system_prompt_template])

        sender = sender_voter_relationship.sender

        if sender.sender_information is None:
            sender.sender_information = {}

        voter = sender_voter_relationship.voter

        if voter.voter_profile is None:
            voter.voter_profile = {}

        self.system_prompt = chat_prompt_template.format(
            candidate_info=sender_voter_relationship.sender.sender_information,
            voter_info=sender_voter_relationship.voter.voter_profile.to_dict()
        )

        super().__init__(self.system_prompt, name, "Handles scheduling and high-level decisions", sender_voter_relationship_id)

        self.conversation_history = initialize_conversation(self.system_prompt)

        first_llm_response = get_llm_response_to_conversation(self.conversation_history)

        self.conversation_history.append(first_llm_response)

        self.available_actions = json.dumps([CreateTextingAgent().to_dict(), CreateRoboCallerAgent().to_dict(), CreateEmailAgent().to_dict()])


        print(f"Created a new PlanningAgent")
        print(f"Initial message: {self.last_message()}")

        db.session.add(self)
        db.session.commit()