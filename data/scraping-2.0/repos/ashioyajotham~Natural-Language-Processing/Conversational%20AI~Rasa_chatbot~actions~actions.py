from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet, EventType
from rasa_sdk.executor import CollectingDispatcher
import requests
import webbrowser
import os
import openai

from dotenv import load_dotenv
import requests
import os

load_dotenv()

# Action to pull the latest news from the web
class NewsAPI(object):
    def __init__(self):
        self.url = 'https://newsapi.org/v2/top-headlines?country=us&apiKey={}'.format(os.getenv("NEWS_API_KEY"))

        self.data = requests.get(self.url).json()
        self.articles = self.data['articles']
        self.news = []
        for article in self.articles:
            self.news.append(article['title'])
        self.news = '\n'.join(self.news)

        print(self.news)

    def get_news(self):
        return self.news
    
class ActionOwner(Action):
    def name(self) -> Text:
        return "action_owner"

    async def run(
        self,
        dispatcher,
        tracker: Tracker,
        domain: "Dict",
    ) -> List[Dict[Text, Any]]:
        url="https://www.linkedin.com/in/ashioyajotham/"
        dispatcher.utter_message("Hold on... Opening my owner's LinkedIn profile.")
        #webbrowser.open(url)
        return []

class ActionOwnerName(Action):
    def name(self) -> Text:
        return "action_owner_name"

    async def run(
        self,
        dispatcher,
        tracker: Tracker,
        domain: "Dict",
    ) -> List[Dict[Text, Any]]:
        dispatcher.utter_message("My owner's name is Victor Ashioya.")
        return []


# Chatgpt -->
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Engine.list()

class ChatGPT(object):

    response = openai.Completion.create(
        model = "text-davinci-003",
        prompt = 'Answer the following question, based on the data shown.'\
        'Answer in a complete sentence.',
        temperature = 0.9,
        max_tokens = 150,
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0.6
    )

    def ask(self, news, question):
        content = news + '\n' + question + '\nAnswer:'

        self.response = openai.Completion.create(
            model = "text-davinci-003",
            prompt = content,
            temperature = 0.9,
            max_tokens = 150,
            top_p = 1,
            frequency_penalty = 0,
            presence_penalty = 0.6
        )
        return self.response['choices'][0]['text']
# Chatgpt <--
chat_api = ChatGPT()
news_api = NewsAPI()

class ActionNews(Action):
    def name(self) -> Text:
        return "action_news"

    async def run(
        self,
        dispatcher,
        tracker: Tracker,
        domain: "Dict",
    ) -> List[Dict[Text, Any]]:
        news = news_api.get_news() # this is the news list which we later pass as a slot
        dispatcher.utter_message(news)
        return [SlotSet("news", news)] #the first news is the slot name and the second is the value
        
# Fetch the question from the user and pass it to the chatbot
class ActionAnswer(Action):
    def name(self) -> Text:
        return "action_chat"

    async def run(
        self,
        dispatcher: CollectingDispatcher, # CollectingDispatcher is used to send messages back to the user
        tracker: Tracker,
        domain: "Dict",
    ) -> List[Dict[Text, Any]]:
        #previous_response = tracker.latest_message['news']
        previous_response = tracker.latest_message['text']
        #question = tracker.latest_message['text']
        question = tracker.latest_message['text']
        answer = chat_api.ask(previous_response, question)
        dispatcher.utter_message(text=answer)
        #return [SlotSet("answer", answer)]
        
# Fetch the question from the user and pass it to the
        #question = tracker.latest_message['text']
        #answer = chat_api.ask(previous_response, question)
        #dispatcher.utter_message(text=answer)
        
# add an utter_default action
class ActionDefaultFallback(Action):
    def name(self) -> Text:
        return "action_default_fallback"
    
    async def run(
        self,
        dispatcher,
        tracker: Tracker,
        domain: "Dict",
    ) -> List[Dict[Text, Any]]:
        dispatcher.utter_message("Sorry, I don't understand. Please try again.")
        return []
