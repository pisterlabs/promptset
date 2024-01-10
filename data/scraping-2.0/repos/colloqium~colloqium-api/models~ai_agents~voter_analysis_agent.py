from models.voter import Voter
from models.interaction import Interaction, SenderVoterRelationship
from models.ai_agents.agent import Agent
from tools.utility import initialize_conversation, get_llm_response_to_conversation
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
import json
from context.database import db

class VoterAnalysisAgent(Agent):
    __mapper_args__ = {
        'polymorphic_identity': 'voter_analysis_agent'
    }

    def __init__(self, voter_id: int, latest_interaction_id: int):
        print(f"Creating a new VoterAnalysisAgent for voter_id {voter_id} and latest_interaction_id {latest_interaction_id}")

        # get voter
        voter = Voter.query.get(voter_id)

        # get latest interaction
        latest_interaction = Interaction.query.get(latest_interaction_id)

        # get sender
        sender = latest_interaction.sender

        # get sender_voter_relationship
        sender_voter_relationship = SenderVoterRelationship.query.filter_by(sender_id=sender.id, voter_id=voter.id).first()

        #Set a prompt for the anlaysis agent to review the voter profile, the latest interaction, and update the voter profile so that we can tailor or contact to the voter and be more responsive to their needs
        prompt_template = '''
            You are an voter analysit for political campaigns. Your job is to keep the voter profile current and relevant for the voter so that our future communications with them are more responsive to their needs and relevant to their lives.
            You're job is to be empathetic and understand the voter's needs and concerns and update the voter profile accordingly.

            You will be given the current voter profile, and a the most recent touch point that the voter had with the campaign.
            and a brief description of how you learned the information about the voter. Meant to be used as context for another agent later. Needs appropriate context to have a consistent conversation with the voter.

            Please output your analysis as a json object in the form:

            {{
                
                "last_interaction": "A description of the voter's last interaction so future conversations can have appropriate context",
                "interests": "A description of the voter's interests",
                "policy_preferenes": "A description of the voter's policy preferences", 
                "preferred_contact_method": "The voter's preferred contact method",
                "background": "Any background information about the voter that will help someone have a more informed conversation later, that is not captured somewhere else in the voter profile"
            }}

        '''

        system_prompt_template = SystemMessagePromptTemplate.from_template(prompt_template)

        chat_prompt_template = ChatPromptTemplate.from_messages([system_prompt_template])

        self.system_prompt = chat_prompt_template.format()

        super().__init__(self.system_prompt, "voter_analysis_agent", "Analyzes voter profile and updates it", sender_voter_relationship.id)

        self.conversation_history = initialize_conversation(self.system_prompt)

        db.session.add(self)
        db.session.commit()