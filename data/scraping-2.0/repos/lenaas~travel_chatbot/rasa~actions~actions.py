# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from rasa_sdk.events import AllSlotsReset
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import os
from dotenv import load_dotenv
import openai 
import json
import numpy as np

map_time = {
    "1. Quarter": ["Jan", "Feb", "Mär"],
    "2. Quarter": ["Apr", "Mai", "Jun"],
    "3. Quarter": ["Jul", "Aug", "Sep"],
    "4. Quarter": ["Okt", "Nov", "Dez"]
}
map_climate = {
    "Warm and sunny": "warm",
    "Cold Weather": "kalt"
}
map_activity = {
    "Relaxing on the beach": "Strandurlaub",
    "Exploring a city": "Städtereise",
    "Experiencing adventures": "Rundreise",
    "Experiencing culture": "Kultur"
}
map_interest = {
    'History': 'Geschichte',
    'Nature': 'Natur',
    'Culture': 'Kultur',
    'Great food': 'Kulinarik',
    'Party': 'Party',
    'Wellness': 'Wellness',
    'Adventure': 'Abenteuer'
}
map_budget = {
    'Lower': 'Günstiger als Deutschland',
    'Equal': 'Durchschnitt Deutschland',
    'Higher': 'Teurer als Deutschland',
}
map_housing = {
    'Camping': 'Camping',
    'Hotel/Hostel/Vacation house': 'Ferienhaus/Hotel/Hostel',
}
map_months = {
    "Jan": "Januar",
    "Feb": "Februar",
    "Mär": "März",
    "Apr": "April",
    "Mai": "Mai",
    "Jun": "Juni",
    "Jul": "Juli",
    "Aug": "August",
    "Sep": "September",
    "Okt": "Oktober",
    "Nov": "November",
    "Dez": "Dezember"
}

class ActionGetDestinations(Action):

    def name(self) -> Text:
        return "action_get_destinations"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # load input
        slot_keys = ["time","climate","activity","interest_1","interest_2","budget","housing","use_GPT"]
        slot_values = [tracker.get_slot(i) for i in slot_keys]
        slot_dict = dict(zip(slot_keys, slot_values))
        
        # load database
        with open("./actions/dataset/dataset_destination.json") as f:
            data = json.load(f)
    
        length=len(data['Reiseziel'])
        
        #Logik für Datenbank
        def easy_count(key,input,mapper,length=length):
            '''checks for each destination if the activity is available and returns a list of 1 and 0'''
            return [1 if mapper[input] in data[key][str(dest)] else 0 for dest in range(length)]
        def weighted_count(key, input,mapper,weight=0.5,length=length):
            '''function to weight the differing interests priorities. Otherwise same count as easy_count'''
            # Interest 1
            int_1 = [1 if mapper[input[0]] in data[key][str(dest)] else 0 for dest in range(length)]
            # Interest 2
            int_2 = [round(weight,1) if mapper[input[1]] in data[key][str(dest)] else 0 for dest in range(length)]
            return np.array(int_1) + np.array(int_2)
        def evaluate_climate(key, input, mapper_a,mapper_b,length=length):
            ''''''
            # get climate score
            # iterate months of quarter
            climate = []
            rain = []
            for month in mapper_a[input[0]]:
                # get bool array with climate hit for each month
                climate.append([1 if mapper_b[input[1]] in data[key][str(dest)][month]['Klima'] else 0 for dest in range(length)])
                # get rain score for each month
                rain.append([data[key][str(dest)][month]['Regenwahrscheinlichkeit']for dest in range(length)])
            climate_sum = np.round(np.array(np.array(climate[0])+np.array(climate[1])+np.array(climate[2]))/3,1)
            rain_sum = np.round(np.array(np.array(rain[0])+np.array(rain[1])+np.array(rain[2]))/3,1)
            # Reverse Rain score - little rain is good
            rain_sum = np.absolute(1 - rain_sum)
            return climate_sum, rain_sum

        def evaluate_time(key, input, mapper_a,mapper_b,length=length):
            # for each month in chosen quarter
            time= []
            for month in mapper_a[input]:
                # check if travel is recommended for that month
                time.append([1 if mapper_b[month] in data[key][str(dest)] else 0 for dest in range(length)])
            return np.round(np.array(np.array(time[0])+np.array(time[1])+np.array(time[2]))/3,1)


        def compute_total():
            activity = np.array(easy_count(key="Reiseart",input=slot_dict['activity'],mapper=map_activity))
            budget = np.array(easy_count(key="Preisniveau",input=slot_dict['budget'],mapper=map_budget))
            housing = np.array(easy_count(key="Unterkunft",input=slot_dict['housing'],mapper=map_housing))
            interest = np.round(np.array(weighted_count(key="Interessen",input=[slot_dict['interest_1'],slot_dict['interest_2']],mapper=map_interest))/1.5,1)
            climate, rain = evaluate_climate('Klima und Regenwahrscheinlichkeit',[slot_dict['time'],slot_dict['climate']],mapper_a=map_time,mapper_b=map_climate)
            time = np.array(evaluate_time("Beste Monate zum Reisen", slot_dict['time'],mapper_a=map_time, mapper_b=map_months))
            assert len(activity) == len(budget) == len(housing) == len(interest) == len(climate) == len(rain) == len(time), "Length of score lists not equal"
            return np.round(np.array(activity + budget + housing + interest + np.array(climate) + np.array(rain) + time)/7,4)

        def get_top5(score_list):
            sorted = np.flip(np.argsort(score_list))
            destinations = [data["Reiseziel"][str(dest)]for dest in sorted[:5]]
            scores = score_list[sorted[:5]]
            return destinations,scores

        dest,scores = get_top5(compute_total())
        output_string= f'- 1. {dest[0]} - Score: {scores[0]} \n' + f'- 2. {dest[1]} - Score: {scores[1]} \n' + f'- 3. {dest[2]} - Score: {scores[2]} \n' + f'- 4. {dest[3]} - Score: {scores[3]} \n' + f'- 5. {dest[4]} - Score: {scores[4]} \n'
        dispatcher.utter_message(text="Thank you for providing all the necessary details. Based on my internal database, , I recommend considering the following travel destinations: \n"+output_string+" If none of these destinations are suitable for you, I can also do a quick internet search based on your criteria.",
                                    buttons= [
            {"payload":'/GPT{"use_GPT":"Yes"}', "title": "Yes, do it!"},
            {"payload":'/GPT{"use_GPT":"No"}', "title": "No, no further help is needed"},
        ])

        return []

class Conduct_GPT_search(Action):
    def name(self) -> Text:
        return "action_conduct_GPT_search"
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
            slot_keys = ["time","climate","activity","interest_1","interest_2","budget","housing","use_GPT"]
            slot_values = [tracker.get_slot(i) for i in slot_keys]
            slot_dict = dict(zip(slot_keys, slot_values))
            if slot_dict["use_GPT"]=="Yes":
                load_dotenv()
                openai.api_key = os.getenv('OPENAI_API_KEY')
                assert openai.api_key!=None, "API_KEY is not set"
                prompt = f"""Imagine you are a Travel Agent. \n
                        I will list you some criteria and you should give me the top 5 travel destinations based on the criteria as a bulletpoint list.\n
                        Keep the answer short and simple. I dont want any explanations concerning the destinations \n
                        Compute me a score with a range of [0-1] for each destination based on the criteria and sort the destinations by the score. \n
                        Return the destinations in the following format: State/City, Country, Score \n
                        1. Travel Time: {slot_dict["time"]}\n
                        2. Climate and Weather: {slot_dict["climate"]}\n
                        3. Activity: {slot_dict["activity"]}\n
                        4. Primary Interest: {slot_dict["interest_1"]}\n
                        5. Secondary Interest: {slot_dict["interest_2"]}\n
                        6. Budget: {slot_dict["budget"]} to Germany\n
                        7. Housing: {slot_dict["housing"]}\n
                        """
                output = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"user",
                            "content": prompt
                        }])
                dispatcher.utter_message(text="I consulted ChatGPT and it recommends the following destinations: \n"+str(output["choices"][0]["message"]["content"])+ "\n I hope you find these recommendations helpful! If you have any other questions or need further assistance, feel free to ask. Enjoy your vacation planning!")
            else:
                dispatcher.utter_message(text="I hope you found what you were looking for. If you need further assistance, please let me know.")
            return []

class Reset_Slots(Action):
    def name(self) -> Text:
        return "action_reset_slots"
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]: 
        return [AllSlotsReset()]
