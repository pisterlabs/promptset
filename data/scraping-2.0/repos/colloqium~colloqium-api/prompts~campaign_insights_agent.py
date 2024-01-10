from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from models.sender import Campaign, Sender
from models.interaction import Interaction, InteractionStatus


def get_campaign_summary_system_prompt(campaign: Campaign):

    # Get an array of all the summary information from each interaction in this campaign
    # relevant fields are: goal_achieved, rating_explanation, rating, campaign_relevance_score, campaign_relevance_explanation, campaign_relevance_summary, insights_about_issues, insights_about_voter
    # filter out interaction statuses that are less than responded
    interactions = Interaction.query.filter(Interaction.campaign_id == campaign.id, Interaction.interaction_status >= InteractionStatus.RESPONDED).all()
    sender = Sender.query.get(campaign.sender_id)

    # Get the relevant fields for all interactions and append them to an array
    interaction_summaries = ""

    for interaction in interactions:
        interaction_string = f"Interaction for voter {interaction.voter.voter_name} - "
        interaction_string += f"Goal achieved: {interaction.goal_achieved} "
        interaction_string += f"Rating explanation: {interaction.rating_explanation} "
        interaction_string += f"Rating: {interaction.rating} "
        interaction_string += f"Campaign relevance score: {interaction.campaign_relevance_score} "
        interaction_string += f"Campaign relevance explanation: {interaction.campaign_relevance_explanation} "
        interaction_string += f"Campaign relevance summary: {interaction.campaign_relevance_summary} "
        interaction_string += f"Insights about issues: {interaction.insights_about_issues} "
        interaction_string += f"Insights about voter: {interaction.insights_about_voter} "
        interaction_string += "\n"
        interaction_summaries += interaction_string


    print (f"Interaction summary: {interaction_summaries}")

    
        

    # GPT API System Prompts
    system_prompt = '''
                You are a senior campaign manager with a comprehensive skill set.

                Your team has just completed a targeted voter outreach. Your role is to synthesize this data into actionable, brief summaries for different roles on the campaign team.

                You will generate:
                1. Policy insights focusing on 1) the most common and 2) the most interesting policy areas. Limit to 5 areas or fewer. Leave empty if no insights specifically from this campaign.
                2. A 15-20 word summary for the Communications Director.
                3. A 15-20 word summary for the Field Director.
                4. A 15-20 word summary for the Campaign Manager, synthesizing all the above.

                Keep summaries actionable, clear, and brief.

                Output your findings in this JSON format:

                {{
                    "policy_insights": {{ "Taxes": "insight", "Abortion": "insight", "Education": "insight" }} //illustrative,
                    "communications_director_summary": "concise insights for comms",
                    "field_director_summary": "concise insights for field",
                    "campaign_manager_summary": "concise, holistic insights"
                }}

                Campaign Context:
                {sender_information}

                Interaction Data:
                {interaction_summaries}

                '''
 

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_prompt)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

    output = chat_prompt.format(sender_information=sender.example_interactions, interaction_summaries=interaction_summaries)

    print(f"Campaign Summary System prompt: {output}")

    return output