
# from typing import Any, Text, Dict, List
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
# from rasa_sdk.events import Restarted, SlotSet
# import pymongo
# from datetime import datetime
# import time
# import sys
# import random
# from bson import ObjectId
# import openai
# client = pymongo.MongoClient("mongodb+srv://chatbot:stark#777@cluster0.ond2rhh.mongodb.net/")
# DataBase = client['GACData']
# coll1 = DataBase['GAC2023']
# openai.api_key = "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"
# class ActionFallback(Action):
#     def name(self) -> Text:
#         return "say_fallback"

#     def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         user_input = tracker.latest_message.get("text")
#         response = openai.Completion.create(
#             engine="text-davinci-003",  # GPT-3.5 engine
#             prompt=user_input,
#             max_tokens=50)

#         fallback_response = response.choices[0].text.strip()
#         dispatcher.utter_message(text=fallback_response)
# class ConvoRestart(Action):
#     def name(self) -> Text:
#         return "restart_convo"
#     def run(self, dispatcher:CollectingDispatcher, tracker:Tracker, domain: Dict[Text,Any]) -> List[Dict[Text,Any]]:
#         return[Restarted()]
# class GAC2023(Action):
#     def name(self) -> Text:
#         return "gac2023_basic_info"
#     def run(self, dispatcher:CollectingDispatcher, tracker:Tracker, domain: Dict[Text,Any]) -> List[Dict[Text,Any]]:
#         DetailType=""
#         entities = tracker.latest_message.get('entities', [])
#         SpecDetails = next((entity for entity in entities if entity['entity'] == 'general_info'), None)
#         if SpecDetails is not None:
#             DetailType = str(SpecDetails['value'])
#         else:
#             dispatcher.utter_message("Sorry, but can you please elaborate the question?")
#         if(DetailType == "L"):
#             ChoiceNum = random.randint(1,3)
#             if(ChoiceNum == 1):
#                 dispatcher.utter_message("The Global Alumni Convention is going to be held in the Multi-Purpose Hall of MNNIT, also known as the MP hall")
#             elif(ChoiceNum == 2):
#                 dispatcher.utter_message("GAC 2023 is going to be held in the institute's MP hall, also called the Multi Purpose hall")
#             else:
#                 dispatcher.utter_message("The Alumni Convention 2023 is going to be conducted in the Multi Purpose hall of MNNIT")
#         elif(DetailType == "DESC"):
#             ChoiceNum = random.randint(1,4)
#             if(ChoiceNum == 1):
#                 doc = coll1.find_one({"id":"DESC"}, {"_id":0})
#                 desc2023 = doc.get("desc1",{})
#                 dispatcher.utter_message(desc2023)
#             elif(ChoiceNum == 2):
#                 doc = coll1.find_one({"id":"DESC"}, {"_id":0})
#                 desc2023 = doc.get("desc2",{})
#                 dispatcher.utter_message(desc2023)
#             else:
#                 doc = coll1.find_one({"id":"DESC"},{"_id":0})
#                 desc2023 = doc.get("desc3",{})
#                 dispatcher.utter_message(desc2023)
#         elif(DetailType == "DL"):
#             ChoiceNum = random.randint(1,3)
#             if(ChoiceNum == 1):
#                 dispatcher.utter_message("GAC 2023 is going to be held in the MP Hall of MNNIT, from 8 a.m November 4(Saturday) to 3 p.m November 5(Sunday)")
#             elif(ChoiceNum == 2):
#                 dispatcher.utter_message("GAC 2023 is scheduled to take place at the MP Hall of MNNIT, and it will run from 8 a.m. on Saturday, November 4th, to 3 p.m. on Sunday, November 5th.")
#             else:
#                 dispatcher.utter_message("Global Alumni Convention 2023 is scheduled to take place at the MP Hall of MNNIT, and it will run from 8 a.m. on Saturday, November 4th, to 3 p.m. on Sunday, November 5th.")
#         elif(DetailType=="S"):
#             ChoiceNum = random.randint(1,3)

#             if(ChoiceNum == 1):
#                 dispatcher.utter_message("Here is the detailed Schedule of GAC 2023: https://vaave.s3.amazonaws.com/attachments/1687341441_677f9c1773f98a424dde65c1c0563f2e.pdf")
            
#             elif(ChoiceNum == 2):
#                 dispatcher.utter_message("You can find the event schedules of GAC 2023 following this link: https://vaave.s3.amazonaws.com/attachments/1687341441_677f9c1773f98a424dde65c1c0563f2e.pdf")
#             else:
#                 dispatcher.utter_message("This is the link to the event schedule page: https://vaave.s3.amazonaws.com/attachments/1687341441_677f9c1773f98a424dde65c1c0563f2e.pdf")
        
#         else:
#             ChoiceNum = random.randint(1,3)
#             if(ChoiceNum == 1):
#                 dispatcher.utter_message("The Global Alumni Convention is going to be held in the Multi-Purpose Hall of MNNIT, also known as the MP hall")
#             elif(ChoiceNum == 2):
#                 dispatcher.utter_message("GAC 2023 is going to be held in the institute's MP hall, also called the Multi Purpose hall")
#             else:
#                 dispatcher.utter_message("The Alumni Convention 2023 is going to be conducted in the Multi Purpose hall of MNNIT")
# class ConvoRestart(Action):
#     def name(self) -> Text:
#         return "gac2023_orgcom"
#     def run(self, dispatcher:CollectingDispatcher, tracker:Tracker, domain: Dict[Text,Any]) -> List[Dict[Text,Any]]:
#         DetailType=""
#         entities = tracker.latest_message.get('entities', [])
#         SpecDetails = next((entity for entity in entities if entity['entity'] == 'gacorg'), None)
#         if SpecDetails is not None:
#             DetailType = str(SpecDetails['value'])
#         else:
#             dispatcher.utter_message("Sorry, but can you please elaborate the question?")
#         identifiers = ["CoChair","oSec","Tres","Jsec","chair","Ad","Pat"]
#         if DetailType not in identifiers:
#             query = {"id":"OrgComm"}
#             doc = coll1.find_one(query,{"_id":0})
#             for i in doc:
#                 dispatcher.utter_message("Here is the list of members from the Organising Committee")
#                 dispatcher.utter_message(i+":"+str(doc.get(i,{})))
#         elif DetailType == "CoChair":
#             query = {"id":"OrgComm"}
#             doc = coll1.find_one(query,{"id":0})
#             Name = doc.get("Co-Chairman",{})
#             dispatcher.utter_message("The Co-Chairman of the event is, "+ str(Name))
#         elif DetailType == "oSec":
#             query = {"id":"OrgComm"}
#             doc = coll1.find_one(query,{"id":0})
#             Name = doc.get("Org.Secretary",{})
#             dispatcher.utter_message("The Organising Secretary of the event is, "+ str(Name))
#         elif DetailType == "Tres":
#             query = {"id":"OrgComm"}
#             doc = coll1.find_one(query,{"id":0})
#             Name = doc.get("Treasurer",{})
#             dispatcher.utter_message("The Treasurer of the event is, "+ str(Name))
#         elif DetailType == "Jsec":
#             query = {"id":"OrgComm"}
#             doc = coll1.find_one(query,{"id":0})
#             Name = doc.get("Joint Org.Secretaries",{})
#             dispatcher.utter_message("The Co-Chairman of the event is, "+ str(Name))
#         elif DetailType == "chair":
#             query = {"id":"OrgComm"}
#             doc = coll1.find_one(query,{"id":0})
#             Name = doc.get("Chairman",{})
#             dispatcher.utter_message("The Chairman of the event is, "+ str(Name))
#         elif DetailType == "Ad":
#             query = {"id":"OrgComm"}
#             doc = coll1.find_one(query,{"id":0})
#             Name = doc.get("Advisor",{})
#             dispatcher.utter_message("The Advisor of the event is, "+ str(Name))
#         elif DetailType == "Pat":
#             query = {"id":"OrgComm"}
#             doc = coll1.find_one(query,{"id":0})
#             Name = doc.get("Patron",{})
#             dispatcher.utter_message("The Patron of the event is, "+ str(Name))
# class ConvoRestart(Action):
#     def name(self) -> Text:
#         return "gac_reg"
#     def run(self, dispatcher:CollectingDispatcher, tracker:Tracker, domain: Dict[Text,Any]) -> List[Dict[Text,Any]]:
#         DetailType=""
#         entities = tracker.latest_message.get('entities', [])
#         SpecDetails = next((entity for entity in entities if entity['entity'] == 'gacreg'), None)
#         if SpecDetails is not None:
#             DetailType = str(SpecDetails['value'])
#         else:
#             dispatcher.utter_message("Sorry, but can you please elaborate the question?")
#         if DetailType == "FR" or DetailType =="RL":
#             ChoiceNum = random.randint(1,4)
#             if ChoiceNum == 1:
#                 dispatcher.utter_message("You can register for GAC 2023, on this link:https://alumni.mnnit.ac.in/easysignup/register/event/278773.dz")
#             elif ChoiceNum == 2:
#                 dispatcher.utter_message("You can register on this link, https://alumni.mnnit.ac.in/easysignup/register/event/278773.dz")
#             else:
#                 dispatcher.utter_message("This is the link for registration, https://alumni.mnnit.ac.in/easysignup/register/event/278773.dz")
#         elif DetailType == "FO":
#             ChoiceNum = random.randint(1,3)
#             if ChoiceNum == 1:
#                 dispatcher.utter_message("The fee amount for non-felicitated batches is,₹5,000 ")
#             elif ChoiceNum == 2:
#                 dispatcher.utter_message("As a token of our appreciation for those not in the felicitated batches, the fee is a humble ₹5,000")
#             else:
#                 dispatcher.utter_message("For our cherished alumni in the non-felicitated batches, the fee is a mere ₹5,000, reflecting our gratitude for your continued support")
#         elif DetailType == "FF":
#             ChoiceNum = random.randint(1,3)
#             if ChoiceNum == 1:
#                 dispatcher.utter_message("The fee amount for felicitated batches is,₹7,500 ")
#             elif ChoiceNum == 2:
#                 dispatcher.utter_message("As a heartfelt gesture to our celebrated alumni in the felicitated batches, the fee is set at ₹7,500, a sincere reflection of our admiration for your outstanding accomplishments")
#             else:
#                 dispatcher.utter_message("For our cherished alumni in the felicitated batches, the fee is a mere ₹7,500, reflecting our gratitude for your continued support")
#         elif DetailType == "F":
#             ChoiceNum = random.randint(1,3)
#             if ChoiceNum == 1:
#                 dispatcher.utter_message("You can find the complete fee details for GAC 2023, a warm and welcoming event, by visiting this link:https://vaave.s3.amazonaws.com/attachments/1694691512_ddad0639613188f276576af5d6e44f35.png")
#             elif ChoiceNum == 2:
#                 dispatcher.utter_message("Discover the full fee details for GAC 2023, a gathering filled with heartwarming moments, at this link: https://vaave.s3.amazonaws.com/attachments/1694691512_ddad0639613188f276576af5d6e44f35.png.")
#             else:
#                 dispatcher.utter_message("The comprehensive fee information for GAC 2023, a celebration of our cherished alumni, can be found here: https://vaave.s3.amazonaws.com/attachments/1694691512_ddad0639613188f276576af5d6e44f35.png")
#         elif DetailType=="RD":
#             ChoiceNum = random.randint(1,3)
#             if ChoiceNum == 1:
#                 dispatcher.utter_message("Our heartwarming journey together has already begun, and we're excited to let you know that registration for this special event is now open. The last date to join us is November 3rd, so please don't miss out on being part of this memorable occasion")
#             elif ChoiceNum == 2:
#                 dispatcher.utter_message("We're thrilled to announce that our registration process is in full swing, and we can't wait to welcome you. The deadline for registration is November 3rd, so we hope to see you there, creating beautiful memories together")
#             else:
#                 dispatcher.utter_message("It's a joyous moment for us as registration for this event has officially started. We're looking forward to having you join us, and remember, the last day to register is November 3rd. Let's make this event an unforgettable experience together.")