# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import json
from annoy import AnnoyIndex
import openai
from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa_sdk.events import FollowupAction
import os
import logging

openai.api_key = os.environ['OPENAI_API_KEY']

class ActionLLMCall(Action):

	def name(self) -> Text:
		return "llm_call"
	
	def run(self, dispatcher: CollectingDispatcher,tracker: Tracker,domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
		 # Load the Annoy index. Setting the embedding length to 1536
		annoy_index = AnnoyIndex(1536)

		annoy_index.load('./actions/RASA_vector_database.ann') 

		# Load the metadata
		with open('./actions/metadata.json', 'r') as f:
			metadata = json.load(f)

		# Query
		query = tracker.latest_message["text"]
		query_embedding = openai.Embedding.create(
		    input=[query],
		    engine="text-embedding-ada-002"
		)
		xq = query_embedding['data'][0]['embedding']
		nns = annoy_index.get_nns_by_vector(xq, 5)  # returns list of IDs of the top 5 nearest neighbors
		contexts = [metadata[str(nn)]['text'] for nn in nns]

		augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query

		primer = f"""You are Q&A bot. A highly intelligent system that answers
		user questions based on the information provided by the user above
		each question. If the information can not be found in the information
		provided by the user you truthfully say "Sorry, I cannot help with that. Please ask me questions about Rasa Open Source"
		"""

		res = openai.ChatCompletion.create(
		    model="gpt-3.5-turbo",
		    messages=[
		        {"role": "system", "content": primer},
		        {"role": "user", "content": augmented_query}
		    ]
		)
		dispatcher.utter_message(text=res['choices'][0]['message']['content'])
		return [FollowupAction(name="ask_follow_up")]

class ActionFollowUp(Action):

	def name(self) -> Text:
		return "ask_follow_up"

	def run(self, dispatcher: CollectingDispatcher,tracker: Tracker,domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
		dispatcher.utter_message("How else may I assist you today?")
		return [FollowupAction(name=ACTION_LISTEN_NAME)]