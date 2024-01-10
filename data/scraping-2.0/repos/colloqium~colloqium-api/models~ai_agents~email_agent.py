from models.ai_agents.agent import Agent
from models.interaction import SenderVoterRelationship
from models.interaction import Interaction, InteractionStatus
from context.database import db
from tools.utility import get_llm_response_to_conversation, initialize_conversation
from tools.vector_store_utility import get_vector_store_results
from tools.ai_functions.alert_campaign_team import AlertCampaignTeam
from tools.ai_functions.end_conversation import EndConversation
from tools.ai_functions.get_candidate_information import GetCandidateInformation
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from context.sockets import socketio
from context.analytics import analytics, EVENT_OPTIONS
import json

class EmailAgent(Agent):
    __mapper_args__ = {
        'polymorphic_identity': 'email_agent'
    }
    
    def __init__(self, interaction_id: int):

        with db.session.no_autoflush:
            print(f"Creating a new EmailAgent for interaction_id {interaction_id}")

            # get interaction
            interaction = Interaction.query.get(interaction_id)

            if interaction is None:
                raise Exception("Interaction not found")
            
            # get campaign
            campaign = interaction.campaign

            # get sender
            sender = interaction.sender

            # get voter
            voter = interaction.voter

            sender_voter_relationship = SenderVoterRelationship.query.filter_by(sender_id=sender.id, voter_id=voter.id).first()

            # look in the vector store for a subset of example interactions based on the campaign prompt
            key_examples = get_vector_store_results(campaign.campaign_prompt, 3, 0.25, {'context': 'sender', 'id': sender.id})

            # get the key_examples["text"] from each example and remove the brackets
            key_examples = [example["text"] for example in key_examples]

            # remove all [ and { }] from the examples
            key_examples = [example.replace("[", "").replace("]", "").replace("{", "").replace("}", "") for example in key_examples]

            prompt_template = '''
                Hey there! You're wriiting emails to {voter_name} on behalf of {sender_name}. The tone? Let's keep it friendly and straightforward—like chatting with a mature friend. Keep it short and sweet. Be as concise as possible. Assume the person reading the email is extremely busy and needs you to get to the point as quickly as possible.

                Campaign Details:
                {campaign_prompt}

                What We're Trying to Achieve:
                {campaign_goal}

                Campaign End Date:
                {campaign_end_date} (Note: that's election day for political races.)

                Sender Information:
                {sender_information}

                Voter Information:
                {voter_information}

                Example Interactions:
                {{
                    Subject: Come Discuss Community Issues
                    Body: <p>Hi FirstName, it's Sarah. How would you like to join us on DATE for an event about community issues? We'd really value your input. The event on the XXth is basically a space for community voices. {sender_name}, who's running for Mayor, will be there to listen and talk solutions. Sound interesting?</p> <p>{sender_name} has a strong background, especially in criminal justice reform. He's got the experience to make a real difference. Interested in meeting him at the event?</p>
                }}

                Here is the type of information you may have about the candidate:
                {example_interactions}
                
                Don't Know the Answer? Point them here: {campaign_fallback}

                IDs You Might Need:
                Campaign ID: {campaign_id}
                Voter ID: {voter_id}
                Sender ID: {sender_id}

                Remember, these are emails. Return your email including the subject line as a json string in the following format:
                {{
                    "subject": "Example Subject",
                    "body": "Body of the email as html goes here"
                }} 

                Be sure to include all of the relevant information for the voter in the first email. Your goal is to get them to open and read it, assume they won't respond and continue the conversation.

                Include any relevant links and a clear call to action. You should only make affirmative statements or use retorical questions. The person getting the email is not going to respond.
                
                Wait for a human go-ahead before sending the first message.
            '''

            system_prompt_template = SystemMessagePromptTemplate.from_template(prompt_template)

            chat_prompt_template = ChatPromptTemplate.from_messages([system_prompt_template])

            self.system_prompt = chat_prompt_template.format(
                voter_name = voter.voter_name,
                campaign_name = campaign.campaign_name,
                campaign_end_date = campaign.campaign_end_date,
                sender_name = sender.sender_name,
                voter_information = voter.voter_profile.to_dict(),
                campaign_prompt = campaign.campaign_prompt,
                sender_information = sender.sender_information,
                campaign_goal = campaign.campaign_goal,
                campaign_fallback = sender.fallback_url,
                example_interactions = key_examples,
                campaign_id = campaign.id,
                voter_id = voter.id,
                sender_id = sender.id
            )

            super().__init__(self.system_prompt, "email_agent", "Writes emails", sender_voter_relationship.id)

            self.conversation_history = initialize_conversation(self.system_prompt)

            first_llm_response = get_llm_response_to_conversation(self.conversation_history)

            while first_llm_response['content'] == self.system_prompt:
                print("The llm did not return a response. Trying again.")
                first_llm_response = get_llm_response_to_conversation(self.conversation_history)

            self.conversation_history.append(first_llm_response)

            print(f"Generated first response for email agent: {first_llm_response}")

            interaction.conversation = self.conversation_history
            interaction.interaction_status = InteractionStatus.INITIALIZED
 
            self.available_actions = json.dumps([AlertCampaignTeam().to_dict(), EndConversation().to_dict(), GetCandidateInformation().to_dict()])
            self.interactions = [interaction]
            
            db.session.add(self)
            db.session.add(interaction)

            # Send a message to all open WebSocket connections with a matching campaign_id
            socketio.emit('interaction_initialized', {'interaction_id': interaction.id, 'campaign_id': interaction.campaign_id}, room=f'subscribe_campaign_initialization_{interaction.campaign_id}')
            socketio.emit('interaction_initialized', {'interaction_id': interaction.id, 'sender_id': interaction.sender_id}, room=f'subscribe_sender_confirmation_{interaction.sender_id}')


            analytics.track(interaction.voter.id, EVENT_OPTIONS.initialized, {
                'sender_id': interaction.sender.id,
                'sender_email': interaction.select_phone_number(),
                'interaction_type': interaction.interaction_type,
                'interaction_id': interaction.id
            })