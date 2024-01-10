from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from typing import List, Dict

def get_conversation_evaluation_system_prompt(conversation: List[Dict]):

    # GPT API System Prompts
    system_prompt = ''' 
                You are about to evaluate a conversation where the assistant's goal was defined by the initial system prompt. Please consider the assistant's effectiveness in achieving this goal based on the conversation that followed. Identify any new information about the voter that was not known at the beginning of the conversation. Return your evaluation as a valid json object in the following format:
                
                {{
                    "insights_about_voter": new information about the voter,
                    "insights_about_issues": What policiy issue areas were discussed in this conversation? What was the sentiment of the voter on these issues? Assume this will later be aggregated across many conversations. If there are no insights about issues return an empty object {{}}.
                    "campaign_insights": What information from this would be helpful for the sender to know? What might this suggest for follow up questions or trends in voter sentiment? Assume this will later be aggregated across many conversations.,
                    "campaign_goal": what was the objective of this conversation
                    "goal_achieved": "True or False depending on if the goal was achived",
                    "rating_explanation": explanation for why the agent deserves the rating taking in to account the goal, new information recieved, and their overall effectiveness
                    "rating": rating from 1 to 10,
                    "campaign_relevance_explanation": explanation for why this conversation is relevant to the campaign staff. For example, this is informatino that is not brought up anywhere else or the voter is a key influencer in the community,
                    "campaign_relevance_score": score from 1 to 100 for how relevant this conversation is to the campaign staff. Will be used to decide which messages to highlight to the campaign,
                    "campaign_relevance_summary": summary of why this conversation is relevant to the campaign staff. Will be used to aggregate relevant information across many conversations,
                }}

                Here is an example evaluation:
                {{
                    'insights_about_voter': "Adrian, the voter, has moved out of Houston but is still interested in supporting John Whitmire's campaign. Adrian showed particular interest in the candidate's stance on education and the HISD takeover.",
                    'insights_about_issues': {{'HISD Takeover': "Adrian was curious about the candidate's position on the HISD takeover, expressing satisfaction with the response that John Whitmire disagrees with the takeover but supports the Texas Education Agency's efforts to improve underperforming schools.", 'Education': "Adrian showed interest in John Whitmire's broader stance on education. The voter appeared appreciative of John's belief in education as a tool to lift people out of poverty and prevent crime."}},
                    'campaign_insights': 'The conversation revealed that Adrian, though no longer based in Houston, is eager to support the Whitmire campaign virtually. This highlights an opportunity for the campaign to leverage out-of-town supporters in their campaign efforts.',
                    'campaign_goal': 'Get the voter to agree to volunteer for the campaign',
                    'goal_achieved': True,
                    'rating_explanation': "The assistant was successful in achieving the goal of getting the voter to agree to volunteer for the campaign. The assistant also effectively and accurately addressed the voter's queries about John Whitmire's stance on the HISD takeover and education.",
                    'rating': 9,
                    'campaign_relevance_explanation': "The conversation is relevant to the campaign staff as it highlights a potential trend in voters who, despite not living in the area, can still be involved in the campaign. The voter's keen interest in the candidate's stance on education should also be noted.",
                    'campaign_relevance_score': 80,
                    'campaign_relevance_summary': "The conversation identifies the potential to involve out-of-town supporters in the campaign and emphasizes the importance of addressing John Whitmire's stance on education to engage voters."
                }}

                The conversation to evaluate is:
                {conversation}'''


    system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_prompt)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

    output = chat_prompt.format(conversation=conversation)

    return output