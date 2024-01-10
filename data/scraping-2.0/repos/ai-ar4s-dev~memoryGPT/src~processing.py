import src.memory as memory
import openai
from src.config import OPENAI_API_KEY
import re

openai.api_key = OPENAI_API_KEY

class Chatbot:
    def __init__(self):
        self.memory = memory.Memory()

    def get_response(self, user_message):
        # Check if the user's input is present in memory and retrieve the response if available
        response_from_memory = self.memory.find_similar_response(user_message)
        if response_from_memory:
            return response_from_memory

        # If the user's input is not in memory, use GPT-3.5 to generate a new response
        bot_response = self.generate_response_gpt(user_message)

        # Save the new interaction to memory
        self.memory.save_interaction(user_message, bot_response)

        return bot_response

    def generate_response_gpt(self, message):

        # OpenAI GPT-3.5 turbo API call to generate a response
        gpt3_response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': 'You are the large language model designed to respond to humans. The following is a snippet of a conversation between a user and a chatbot'},
                {'role': 'user', 'content': message},
                {'role': 'assistant', 'content': None},
            ]
        )

        # Extract the generated response from GPT-3.5
        response = gpt3_response.choices[0].message['content']

        return response

    #Get the embedding of a given text using the OpenAI API
    def get_embedding(self, text):
        return openai.Embedding.create(input=[text], model = 'text-embedding-ada-002')['data'][0]['embedding']

    #Assess the importance of a chat interaction using the OpenAI GPT-3.5 turbo model
    def get_importance_interaction(self, message, response):
        importance_response = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo',
            messages = [
                {'role': 'system', 'content': 'You are the large language model designed to respond to humans. The following is a snippet of a conversation between a user and a chatbot'},
                {'role': 'user', 'content': message},
                {'role': 'assistant', 'content': response},
                {'role': 'system', 'content': 'Please rate the importance of remembering the above interaction on a scale from 1 to 10 where 1 is trivial and 10 is very important. Only respond with the number, do not add any commentary.'},
            ],
            temperature=0,
            n=1,
            max_tokens=100
        )

        numbers = re.findall(r'\b(?:10|[1-9])\b', importance_response.choices[0].message.content)
        if numbers:
            return int(numbers[0]) / 10

        print("Error: Could not parse importance of interaction. Defaulting to 3 out of 10.")
        return 0.3

    #Assess the importance of a chat insight using the OpenAI GPT-3.5 turbo model
    def get_importance_of_insight(self, insight):
        importance_response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': 'You are a large language model. The following is an insight you gained from of a conversation with a user.'}, 
                {'role': 'assistant', 'content': insight},
                {'role': 'system', 'content': 'Please rate the importance of remembering the above insight on a scale from 1 to 10 where 1 is trivial and 10 is very important. Only respond with the number, do not add any commentary.'}
            ],
            temperature=0,
            n=1,
            max_tokens=100
        )

        numbers = re.findall(r'\b(?:10|[1-9])\b', importance_response.choices[0].message.content)
        if numbers:
            return int(numbers[0]) / 10

        print("Error: Could not parse importance of interaction. Defaulting to 3 out of 10.")
        return 0.3

    # Extract the insights from the conversation
    def get_insights(self, messages):
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages + [{'role': 'system', 'content': 'Please list up to 5 high-level insights you can infer from the above conversation. You must respond in a list format with each insight surrounded by quotes, e.g. ["The user seems...", "The user likes...", "The user is...", ...]'}],
            temperature=0.7,
            n=1,
            max_tokens=500
        )

        insights_list = re.findall(r'"(.*?)"', response.choices[0].message.content)

        insights = []

        for insight in insights_list:
            insights.append({
                'content': insight,
                'importance': self.get_importance_of_insight(insight)
            })

        return insights