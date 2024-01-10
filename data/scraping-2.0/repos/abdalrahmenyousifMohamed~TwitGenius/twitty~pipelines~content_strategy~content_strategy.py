import os
import pandas as pd
import logging , argparse
import subprocess
from preprocessing import main
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

class TwitterContentStrategist:
    def __init__(self, openai_api_key):
        self.llm = ChatOpenAI(temperature=0.5, openai_api_key=openai_api_key, model_name='gpt-3.5-turbo')
        self.logger = logging.getLogger(__name__)
        
    def generate_response(self, mentioned_conversation_tweet_text):
        system_template = """
        You are an AI content strategist with an in-depth understanding of Twitter's ever-evolving landscape.
        Your mission is to provide users with content topics that not only align with their preferences but also leverage current events and trending subjects.
        
        % REQUIREMENT:
        - Do not include answers similar to: "This tweet will not only engage others who are interested in climate action but also invite them to share their favorite eco-artists or exhibits, fostering a sense of community and collaboration. Let's keep the conversation going!"

        % RESPONSE TONE:
        - IMPORTANT YOU SHOULD TREAT AS REAL HUMAN
        - Your responses should exude expertise and foresight.
        - Engage users with insightful, thought-provoking, and relevant content suggestions.
        - Maintain a friendly and conversational tone throughout.

        % RESPONSE FORMAT:

        - Offer a range of content topic suggestions that cater to diverse interests.
        - If a user shows interest in a specific topic, craft a tweet or a thread that delves deep into that subject.
        - Provide clear and concise insights within the character limit.

        % RESPONSE CONTENT:

        - Analyze the user's tweet_text for cues about their interests and concerns.
        - Stay updated with trending Twitter topics, current events, and conversations.
        - Suggest content topics that bridge the user's existing interests with real-time discussions.
        - Draft compelling tweets or threads that encourage user engagement and conversation.

        
        Example:

        If the user's tweet_text mentions "space exploration," you can respond with:

        " Space exploration is an endlessly fascinating topic! Let's discuss the latest discoveries on Mars, the potential for future moon missions, and the role of private companies in the space race. I'll craft a tweet that sparks curiosity and invites others to join the cosmic conversation. Ready for takeoff?"

        Remember to stay ahead of the curve and offer content suggestions that keep users engaged and informed in the dynamic world of Twitter.
        """
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        
        final_prompt = chat_prompt.format_prompt(text=mentioned_conversation_tweet_text).to_messages()
        response = self.llm(final_prompt).content

        self.logger.info("Generated response: %s", response) 
        
        return response

if __name__ == "__main__":
    os.environ['OPENAI_API_KEY'] = ''
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    logging.basicConfig(level=logging.INFO)
    strategist = TwitterContentStrategist(openai_api_key=OPENAI_API_KEY)
    # TECHNIQUE NOT IMPORTANT
    # parser = argparse.ArgumentParser(description='Preprocess CSV data')
    # parser.add_argument('file_path', type=str, help='Path to the CSV file containing the data')
    # args = parser.parse_args()
    
    df = pd.read_csv('../data/cleaned_data.csv')
    value = df.iloc[5]['content']
    logging.info('original content' , value)
    draft_tweets = strategist.generate_response(value)
