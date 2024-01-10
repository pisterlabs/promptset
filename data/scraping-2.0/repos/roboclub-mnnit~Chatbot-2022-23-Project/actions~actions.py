# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet, EventType
from rasa_sdk.executor import CollectingDispatcher, Action
import webbrowser
from rasa_sdk.interfaces import Action
from rasa_sdk.events import SlotSet ,EventType
# from rasa_sdk.forms import FormValidationAction
from rasa_sdk.events import AllSlotsReset, SlotSet
from rasa_sdk.events import UserUtteranceReverted
from rasa_sdk.executor import CollectingDispatcher , Action
from rasa_sdk.types import DomainDict
from mysql_connectivity import DataUpdate
import spacy
import openai
from dotenv import load_dotenv
load_dotenv()
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials

class ActionTellName(Action):

    def name(self) -> Text:
        return "action_tell_name"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        name = tracker.get_slot("name")
        message = " Hi {}! , what is your mobile number ? ".format(name)
        print (message)
        dispatcher.utter_message(text=message)
        return []

class ActionTellName(Action):

    def name(self) -> Text:
        return "action_tell_number"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        number = tracker.get_slot("number")
      #  - text : " Hi {name}! , what is your mobile number 📱 ? "
        message = " Yeah , Your mobile number   is {} . Thanks for giving information ".format(number)
        print (message)
        dispatcher.utter_message(text=message)
        return []


# class ValidateHealthForm(FormValidationAction):

#   def name(self) -> Text:
#       return "validate_health_form"

#   async def required_slots(
#     self,
#     slots_mapped_in_domain: List[Text],
#     dispatcher: "CollectingDispatcher",
#     tracker: "Tracker",
#     domain: "DomainDict"
#   ) -> List[Text]:
#     if tracker.get_slot("confirm_exercise") == True:
#       return ["confirm_exercise", "exercise", "sleep", "diet", "stress", "goal"]
#     else:
#       return ["confirm_exercise", "sleep", "diet", "stress", "goal"]

class ActionVideo(Action):
    def name(self) -> Text:
        return "action_video"

    async def run(self, dispatcher,
        tracker: Tracker,
        domain: Dict) -> List[Dict[Text, Any]]:
        video_url="https://youtu.be/j76Z57O0Acw"
        # dispatcher.utter_message(text="wait... Playing your video.")
        # webbrowser.open(video_url)
        dispatcher.utter_message(text=f"Sure! here's the video link for you. You can watch it [here]({video_url}).")
        return []

class ActionSubmit(Action):
    def name(self) -> Text:
        return "action_submit"

    def run(
        self,
        dispatcher,
        tracker: Tracker,
        domain: Dict) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(template="utter_details_thanks", Name=tracker.get_slot("name"),Mobile_number=tracker.get_slot("number"))
        
        
class ActionImage(Action):
    def name(self) -> Text:
        return 'action_image'

    async def run(
        self,
        dispatcher,
        tracker: Tracker,
        domain: Dict) -> List[Dict[Text, Any]]:
        image_url="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.shiksha.com%2Funiversity%2Fmnnit-allahabad-motilal-nehru-national-institute-of-technology-24357&psig=AOvVaw2Huxtg15KnizdKpPlfMjVG&ust=1679021555456000&source=images&cd=vfe&ved=0CA8QjRxqFwoTCICx0ea43_0CFQAAAAAdAAAAABAE"
        # dispatcher.utter_message(text="Please.... wait image is upload..")
        # webbrowser.open(image_url)
        dispatcher.utter_message(text=f"Sure! here's the image link for you. You can watch it [here]({image_url}).")
        return []
    
class ActionGreet(Action):
    def name(self) -> Text:
        return 'action_greet'
    
    def run(self,dispatcher,tracker,domain):
        dispatcher.utter_message(text="MNNIT is the great campus")
        return []
    
class ActionSubmitProject(Action):
    def name(self) -> Text:
        return "action_submitregister"

    def run(
        self,
        dispatcher,
        tracker: Tracker,
        domain: "DomainDict",
    ) -> List[Dict[Text, Any]]:
	
        user_name = tracker.get_slot("registeremail")
        print("email id  is  : ",user_name) 
        
		
        dispatcher.utter_message(template="utter_details_thanks")
        return []

class sdbndnsdjvs(Action):
    def name(self) -> Text:
        return "action_fallback"
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        scopes={
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
        }
        nlp = spacy.load("en_core_web_md")
        user_input = tracker.latest_message.get("text")
        creds=ServiceAccountCredentials.from_json_keyfile_name("C:\\rasa2.o\\botrushchatbot-59e97e76d71a.json",scopes=scopes)
        file=gspread.authorize(creds)
        workbook=file.open("Final Evaluation Schedule(Match and engage)")
        sheet=workbook.sheet1
        # print(sheet.range('B5:B12'))
        data = sheet.get_all_values()
        # print(data)
        team_schedule = {}
        for row in data[4:]:
            team_name = row[1]
            team_time = row[2]
            team_name = team_name.lower()
            team_schedule[team_name.rstrip()]= team_time
        userstring = tracker.get_slot('team')
        if(userstring!=None):
            userstring = userstring.lower()
            f=user_input.split()
            print("this is running")
            for i in f:
                if(userstring==i.lower()):
                    if userstring in team_schedule.keys():
                        response = f"The schedule for {userstring} is at {team_schedule[userstring]}."
                    else:
                        response = f"Sorry, I don't have the schedule information for {userstring}."
                    dispatcher.utter_message(response)
                    return[]
        print(type(userstring))
        print(userstring)
        team_schedule = {}
        for row in data[4:]:
            team_name = row[1]
            team_time = row[2]
            team_name = team_name.lower()
            team_schedule[team_name.rstrip()]= team_time
        try:
            s=len(user_input.split())
            k=user_input.split()
            t=len(user_input)
            ban=['MNNIT','mnnit']
            for i in range(0,s):
                for j in range(0,2):
                    if(i!=(s-1) and (k[i]).lower()=='bot'):
                        if((k[i+1]).lower()=='rush'):
                            dispatcher.utter_message("Not adequate info provided, try again with more description")
                            return[]
                    if((k[i]).lower()==(ban[j]).lower()):
                        dispatcher.utter_message("Not adequate info provided, try again with more description")
                        return[]
            if(s<3 or t<5):
                dispatcher.utter_message("Not adequate info provided, try again with more description")
                return[]
            doc = nlp(user_input)
            openai.api_key=os.getenv('api')
            completions=openai.Completion.create(engine='text-davinci-002',prompt=user_input,max_tokens=150)
            message=completions.choices[0].text
            answer=message
            print(answer)
            dispatcher.utter_message(answer)
            DataUpdate(user_input,answer)
            print(user_input)
            return[]
        except:
            dispatcher.utter_message("Can't process this information right now.")
            return[]
    
class ActionOpenProjectLinkssbgh(Action):
    def name(self) -> Text:
        return "action_open_project_link_ssb_github"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Replace the `project_link` with the actual link to your self-balancing bot project
        project_link = "https://github.com/roboclub-mnnit/SelfBalancingBot-2022-23-Project"
        webbrowser.open(project_link)
        dispatcher.utter_message(text=f"You can find the github repository  at this link: {project_link}")
        return []
    
class ActionOpenVideoLink1(Action):
    def name(self) -> Text:
        return "action_open_ssb_video_link"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Extract the video link from the tracker
        video_link = "https://youtu.be/lqwXdpySzbQ"

        if video_link:
            # Open the video link in a web browser
            # webbrowser.open(video_link)

            # Send a message to the user acknowledging the action
            dispatcher.utter_message(text=f"Sure! here's the video link for you. You can watch it [here]({video_link}).")
        else:
            # If video link is not available, send a message to the user
            dispatcher.utter_message(text="Sorry, I couldn't find the video link.")

        return []
    
class ActionOpenVideoLink2(Action):
    def name(self) -> Text:
        return "action_open_mcg_video_link"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Extract the video link from the tracker
        video_link = "https://youtu.be/1MGkDt1iHKA"

        if video_link:
            # Open the video link in a web browser
            # webbrowser.open(video_link)

            # Send a message to the user acknowledging the action
            dispatcher.utter_message(text=f"Sure! here's opened the video link for you. You can watch it [here]({video_link}).")
        else:
            # If video link is not available, send a message to the user
            dispatcher.utter_message(text="Sorry, I couldn't find the video link.")

        return []
    
class ActionOpenVideoLink3(Action):
    def name(self) -> Text:
        return "action_open_rcs_video_link"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Extract the video link from the tracker
        video_link = "https://youtu.be/FHDl_1TUIMY"

        if video_link:
            # Open the video link in a web browser
            # webbrowser.open(video_link)

            # Send a message to the user acknowledging the action
            dispatcher.utter_message(text=f"Sure! here's the video link for you. You can watch it [here]({video_link}).")
        else:
            # If video link is not available, send a message to the user
            dispatcher.utter_message(text="Sorry, I couldn't find the video link.")

        return []
    
class ActionOpenVideoLink4(Action):
    def name(self) -> Text:
        return "action_open_ats_video_link"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Extract the video link from the tracker
        video_link = ""

        if video_link:
            # Open the video link in a web browser
            # webbrowser.open(video_link)

            # Send a message to the user acknowledging the action
            dispatcher.utter_message(text=f"Sure! here's the video link for you. You can watch it [here]({video_link}).")
        else:
            # If video link is not available, send a message to the user
            dispatcher.utter_message(text="Sorry, I couldn't find the video link.")

        return []
    
class ActionOpenVideoLink5(Action):
    def name(self) -> Text:
        return "action_open_vtt_video_link"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Extract the video link from the tracker
        video_link = "https://youtu.be/0K2zxtrIdmU"

        if video_link:
            # Open the video link in a web browser
            # webbrowser.open(video_link)

            # Send a message to the user acknowledging the action
            dispatcher.utter_message(text=f"Sure! here's the video link for you. You can watch it [here]({video_link}).")
        else:
            # If video link is not available, send a message to the user
            dispatcher.utter_message(text="Sorry, I couldn't find the video link.")

        return []
    
class DisplaywebAction(Action):
    def name(self) -> Text:
        return "action_gsheet"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        scopes={
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
        }
        creds=ServiceAccountCredentials.from_json_keyfile_name("C:\\rasa2.o\\botrushchatbot-59e97e76d71a.json",scopes=scopes)

        file=gspread.authorize(creds)
        workbook=file.open("Final Evaluation Schedule(Match and engage)")
        sheet=workbook.sheet1
        # print(sheet.range('B5:B12'))
        data = sheet.get_all_values()
        # print(data)
        user_team = tracker.get_slot('team')
        user_team = user_team.lower()
        print(user_team)
        team_schedule = {}
        for row in data[4:]:
            team_name = row[1]
            team_time = row[2]
            team_name = team_name.lower()
            team_schedule[team_name.rstrip()]= team_time

        if user_team in team_schedule.keys():
            response = f"The schedule for {user_team} is at {team_schedule[user_team]}."
        else:
            response = f"Sorry, I don't have the schedule information for {user_team}."

        print(response)

        dispatcher.utter_message(response)
        return[]
    
class InquireEvent(Action):
    def name(self) -> Text:
        return "action_inquire_event"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict) -> List[Dict[Text, Any]]:
        # Get the time entity extracted from user message
        user_team = tracker.get_slot('time')
        print(user_team)
        scopes={
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
        }
        creds=ServiceAccountCredentials.from_json_keyfile_name("C:\\rasa2.o\\botrushchatbot-59e97e76d71a.json",scopes=scopes)

        file=gspread.authorize(creds)
        workbook=file.open("Final Evaluation Schedule(Match and engage)")
        sheet=workbook.sheet1
        # print(sheet.range('B5:B12'))
        data =sheet.get_all_values()
        # print(data)
        team_schedule = {}
        t=""
        for row in data[4:]:
            team_name = row[1]
            team_time = row[2]
            print(team_time)
            team_name = team_name.lower()
            team_schedule[team_name.rstrip()]=team_time
            if(user_team==team_time):
                t=team_name

        print(t)
        if user_team in team_schedule.values():
            response = f"The team scheduled for {user_team} is {t} ."
        else:
            response = f"Sorry, I don't have the schedule information for {user_team}."

        dispatcher.utter_message(response)
        return[]
    
class ActionSubmitProject(Action):
    def name(self) -> Text:
        return "action_submitregister"

    def run(
        self,
        dispatcher,
        tracker: Tracker,
        domain: "DomainDict",
    ) -> List[Dict[Text, Any]]:
	
        user_name = tracker.get_slot("registeremail")
        print("email id  is  : ",user_name) 
        
		
        dispatcher.utter_message(template="utter_details_thanks")
        return []
    
class ActionSubmit(Action):
    def name(self) -> Text:
        return "action_submit"

    def run(
        self,
        dispatcher,
        tracker: Tracker,
        domain: Dict) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(template="utter_details_thanks", Name=tracker.get_slot("name"),Mobile_number=tracker.get_slot("number"))
        
        
class ActionImage(Action):
    def name(self) -> Text:
        return 'action_image'

    async def run(
        self,
        dispatcher,
        tracker: Tracker,
        domain: Dict) -> List[Dict[Text, Any]]:
        image_url="https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.shiksha.com%2Funiversity%2Fmnnit-allahabad-motilal-nehru-national-institute-of-technology-24357&psig=AOvVaw2Huxtg15KnizdKpPlfMjVG&ust=1679021555456000&source=images&cd=vfe&ved=0CA8QjRxqFwoTCICx0ea43_0CFQAAAAAdAAAAABAE"
        dispatcher.utter_message(text=f"Sure! here's the image link for you. You can watch it [here]({image_url}).")
        # webbrowser.open(image_url)
        return []
    
class ActionGreet(Action):
    def name(self) -> Text:
        return 'action_greet'
    
    def run(self,dispatcher,tracker,domain):
        dispatcher.utter_message(text="MNNIT is the great campus")
        return []
    
class ActionVideo(Action):
    def name(self) -> Text:
        return "action_video"

    async def run(self, dispatcher,
        tracker: Tracker,
        domain: Dict) -> List[Dict[Text, Any]]:
        video_url="https://youtu.be/j76Z57O0Acw"
        dispatcher.utter_message(text="wait... Playing your video.")
        webbrowser.open(video_url)
        return []
    
class ActionOpenVideoLink5(Action):
    def name(self) -> Text:
        return "action_idk"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Extract the video link from the tracker
        video_link = "https://youtu.be/0K2zxtrIdmU"

        if video_link:
            # Open the video link in a web browser
            # webbrowser.open(video_link)

            # Send a message to the user acknowledging the action
            dispatcher.utter_message(text=f"Sure! here's the video link for you. You can watch it [here]({video_link}).")
        else:
            # If video link is not available, send a message to the user
            dispatcher.utter_message(text="Sorry, I couldn't find the video link.")

        return []
