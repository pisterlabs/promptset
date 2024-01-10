import json
import openai
from datetime import datetime
import os

MODEL_NAME = "gpt-3.5-turbo"
MODEL_TEMPERATURE = 0.9


class InsightManager:
    def __init__(self, api_key: str, chatbot_name: str, user_total_characters: int, chatbot_total_words: int, messages):
        """ Create an insight manager to generate conversational insights
        :param api_key: OpenAI API Key
        :param chatbot_name: Name of the chatbot
        :param user_total_characters: Total number of characters typed by the user
        :param chatbot_total_words: Total number of words used by the chatbot
        :param messages: List of messages in the conversation
        """
        openai.api_key = api_key
        self.chatbot_name = chatbot_name
        self.user_total_characters = user_total_characters
        self.chatbot_total_words = chatbot_total_words
        self.messages = messages

        timestamp, conversational_data = self._generate_insights()
        self._save_insights(timestamp, conversational_data)

    def _generate_insights(self):
        """ Generate conversational insights
        :return: Timestamp of the conversation, and a dictionary containing the conversational insights"""
        self.messages.append(
            {"role": "system", "content": "Generate a python formatted dictionary containing the topic of our "
                                          "conversation based on the users input, and if it cannot be determined return UNKNOWN, with its key being 'topic'. Additionally, the name of the user if it can "
                                          "be determined, if not return UNKNOWN, with its key being 'user_name'. Send only the dictionary as a string."})

        conversation = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=self.messages,
            temperature=MODEL_TEMPERATURE)

        # Try get response and convert to dictionary
        response = conversation.choices[0].message.content
        # response = eval(response)

        try:
            response_dict = json.loads(response)
            topic = response_dict.get('topic')
            user_name = response_dict.get('user_name')
        except:
            topic = "UNKNOWN"
            user_name = "UNKNOWN"

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        conversation_meta_information = {
            "Name of chatbot": self.chatbot_name,
            "Number of characters typed by user": self.user_total_characters,
            "Number of words used by chatbot": self.chatbot_total_words,
            "Subject of conversation": topic,
            "Name of user": user_name}

        conversation_data = {
            "meta-data": conversation_meta_information,
            "messages": self.messages[1:len(self.messages) - 1]}

        return timestamp, conversation_data

    def _save_insights(self, timestamp, conversation_data):
        """ Save conversational insights to a json file
        :param timestamp: Timestamp of the conversation
        :param conversation_data: Dictionary containing the necessary insight data
        """
        if not os.path.exists('./conversations'):
            os.makedirs('conversations')

        filename = f'conversations/conversation_{timestamp}.json'

        with open(filename, 'w') as file:
            json.dump(conversation_data, file, indent=4)
