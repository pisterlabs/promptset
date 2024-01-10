# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

# Rasa SDK Items
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import AllSlotsReset
from rasa_sdk.events import SlotSet

# Action Server Config Items
import yaml

# Search Items
from elasticsearch import Elasticsearch
import openai

# Part Items
import pandas as pd

try:
    with open('actions/actionConfig.yml', 'r') as f:
        yamldata = yaml.safe_load(f)
except:
    print('WARN - Error loading the configuration file')
    exit()
    
try:
    with open('actions/actionSecrets.yml', 'r') as f:
        yamlsecrets = yaml.safe_load(f)
except:
    print('WARN - Error loading the secrets file')
    exit()

# Setup the part data
useacronyms = False
try:
    # Load the JSON file into a pandas DataFrame
    df = pd.read_json('index/acronyms.json')
    # Convert the strings in the 'Entity' column to lowercase
    df['Acronym'] = df['Acronym'].str.lower()

    # Print the DataFrame
    print("Acronyms Loaded")
    useacronyms = True
except:
    useacronyms = False
    print("WARN - No Acronym Data Found")
    
# Setup Search
numberSearchResults = yamldata['num_search_results']
useElastic = False
yamlindex = yamldata['index_name']
elastic_host = yamldata['elastic_host']
elastic_user = yamlsecrets['elastic_user']
elastic_pw = yamlsecrets['elastic_password']

try:
    # Connect to the Elasticsearch instance
    es = Elasticsearch(hosts=[elastic_host], http_auth=(elastic_user, elastic_pw), verify_certs=False)
    if es.indices.exists(index=yamlindex):
        print("Elastic Connected & Search Index Available")
        useElastic = True
    else:
        print("Elastic Connected but no index exists")
        useElastic = False
except:
    # Handle connectivity issues
    useElastic = False
    print("WARN - Error with Elastic Configuration")

# Setup for openai access
## If not setup, set global boolean to prevent errors
key = str(yamlsecrets['openai_api_key']).strip()
useopenai = ""
if (key is None):
    useopenai = False
    print("WARN - No OPENAI API Key Found - All external queries will be stopped")
else:
    override = yamldata['use_openai']
    if (override is None or override is True):
        openai.api_key = key
        useopenai = True
        print("Using OpenAI")
    else:
        useopenai = False
        print("WARN: Global override set, turning off OpenAI access")

class ActionSendAIGen(Action):
    
    def name(self):
        return "action_send_ai_gen"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        prompt = tracker.get_slot("prompt")
        print(prompt)
        
        text = "Prompt: " + str(prompt)
        dispatcher.utter_message(text=text)
        
        if (useopenai and prompt is not None):
            init="I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with Unknown.\nQ: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: Who was president of the United States in 1955\nA: Dwight D. Eisenhower was president of the United States in 1955.\n\nQ: Which party did he belong to?\nA: He belonged to the Republican Party.\n\nQ: What is the square root of banana?\nA: Unknown\n\nQ: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\nQ: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: How many squigs are in a bonk?\nA: Unknown\n\nQ: "
            fin = "?\nA:"
            dispatcher.utter_message(text="Looking up this question")
            
            total = init + prompt + fin
 
            
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=total,
                temperature=0.05,
                max_tokens=100,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["\n"]
            )
            
            text = response['choices'][0]['text']
            
            if(text=="Unknown"):
                text = "Sorry, I don't know the answer to this one"
            
            dispatcher.utter_message(text=text)
        else:
            print(prompt)
            dispatcher.utter_message(text="Access to this information has not been setup properly. Sorry")
        
        return []

class ActionResetAllSlots(Action):

    def name(self):
        return "action_reset_all_slots"

    def run(self, dispatcher, tracker, domain):
        return [AllSlotsReset()]

class ActionResetSearchSlot(Action):
    
    def name(self) -> Text:
        return "action_reset_search_slot"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return [SlotSet("query", None), SlotSet("project", None)]

class ActionResetPartSlot(Action):
    
    def name(self) -> Text:
        return "action_reset_part_slot"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return [SlotSet("part", None)]
    
class ActionResetPromptSlot(Action):
    
    def name(self) -> Text:
        return "action_reset_prompt_slot"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return [SlotSet("prompt", None)]
        
class ActionPerformSearch(Action):

    def name(self) -> Text:
        return "action_perform_search"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Performing Search...")
        
        query = tracker.get_slot('query')
        project = tracker.get_slot('project')
        
        text = "Searching with Query: " + str(query) + " Project: " + str(project)
        dispatcher.utter_message(text=text)
        
        queryPopulated = True
        if(query is None or query == ""):
            queryPopulated = False
            text = "No query was passed. Sorry!!"
            dispatcher.utter_message(text=text)
            
        if(useElastic and queryPopulated):
        # Only entered if upfront items work
            if (project is None or project == ""):
                query_body1 = {
                    "match": {
                        'text': {
                            'query': query,
                            'minimum_should_match': '100%'
                        }
                    }
                }
                
                query_body2 = {
                    "match": {
                        'text': {
                            'query': query,
                        }
                    }
                }
                
                results1 = es.search(index=yamlindex, query=query_body1, sort= [{'_score': {'order': 'desc'}}], size=numberSearchResults)
                
                text = "Here's the top results:"
                dispatcher.utter_message(text=text)
                
                #Parse results
                names1 = [docu['_source']['url'] for docu in results1['hits']['hits']]
                scores1 = [docu['_score'] for docu in results1['hits']['hits']]
                i = 0
                
                if (names1 is None):
                    # Try the other search
                    results2 = es.search(index=yamlindex, query=query_body2, sort= [{'_score': {'order': 'desc'}}], size=numberSearchResults)
                    
                    #Parse results
                    names2 = [docu['_source']['url'] for docu in results2['hits']['hits']]
                    scores2 = [docu['_score'] for docu in results2['hits']['hits']]
                    i = 0
                    
                    if (names2 is None):
                        # Handle empty set
                        text = "No Results Found."
                        dispatcher.utter_message(text=text)
                    else:
                        score2 = scores2[i]
                        text = "[" + name + "](" + name + ") and score " + str(score2) + " - Partial Match"
                        dispatcher.utter_message(text=text)
                        i = i+1
                else:
                    for name in names1:
                        score1 = scores1[i]
                        text = "[" + name + "](" + name + ") and score " + str(score1) + " - Exact Match"
                        dispatcher.utter_message(text=text)
                        i = i+1
                        
                    if (i<numberSearchResults):
                        # Didn't get three exact matches
                        results2 = es.search(index=yamlindex, query=query_body2, sort= [{'_score': {'order': 'desc'}}], size=numberSearchResults)
                        
                        #Parse results
                        names2 = [docu['_source']['url'] for docu in results2['hits']['hits']]
                        scores2 = [docu['_score'] for docu in results2['hits']['hits']]
                        i = 0
                        
                        if (names2 is None):
                            text = "Only Select Results Found."
                            dispatcher.utter_message(text=text)
                        else:
                            for name in names2:
                                if (name in names1):
                                    # Skip duplicate results
                                    i = i + 1
                                else:
                                    score2 = scores2[i]
                                    text = "[" + name + "](" + name + ") and score " + str(score2) + " - Partial Match"
                                    dispatcher.utter_message(text=text)
                                    i = i+1
            else:
                # Project is populated
                
                # Code needs to be written
                text = "Not sure how we got here but this is still under construction"
                dispatcher.utter_message(text=text)
        else:
            if (useElastic is False):
                text = "Search not correctly configured. Sorry!!"
                dispatcher.utter_message(text=text)
            else:
                text = "Query was passed blank. No search is possible. Sorry."
                dispatcher.utter_message(text=text)

        return []
    
class ActionLookupPart(Action):

    def name(self) -> Text:
        return "action_lookup_part"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        print("In Lookup Part")
        dispatcher.utter_message(text="Looking it up...")
        
        part = tracker.get_slot('part')
        text = "Got it! Part: " + str(part)
        dispatcher.utter_message(text=text)
        
        print(str(part))
        
        numResponses = 0
        
        entityFilled = True
        
        if (part is None or part == ""):
            entityFilled = False
            text = "Passed search string was blank. Sorry!!"
            dispatcher.utter_message(text=text)

        if (useacronyms or entityFilled is False):
            # Find the entity in the 'Acronym' column
            entity = part

            # Wrap a try around the lookup in case the acronym is not found
            try:
                # Return the item in the 'Acronym' column
                item = df.loc[df['Acronym'] == entity.lower(), 'Description'].values[0]

                #print(item)  # Output: 'Item 1'
                
                # Create the response
                text = str(entity) + " stands for " + str(item)
                numResponses = numResponses + 1
                dispatcher.utter_message(text=text)
            except:
                print("No Acronym Match Found")
        
        if (numResponses == 0):
            text = "After reviewing the documentation, no information was found. Sorry."
            dispatcher.utter_message(text=text)
        
        return []


