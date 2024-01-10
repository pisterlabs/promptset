import os
import openai
import random
import pandas as pd
from dotenv import load_dotenv
from rasa_sdk import Action, Tracker
from typing import Any, Text, Dict, List
from rasa_sdk.executor import CollectingDispatcher
from google.oauth2 import service_account
from oauth2client.service_account import ServiceAccountCredentials
import gspread
load_dotenv()

class SimpleGoogleSheetOrChatGPTAction(Action):
    
    def name(self) -> Text:
        return "simple_google_sheet_or_chatgpt_action"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_text = tracker.latest_message.get('text')
        intent = tracker.latest_message.get('intent').get('name')
        entities = tracker.latest_message.get('entities')
        if intent == "customer_details":
            response = self.get_customer_details(entities)
        elif intent == "product_details":
            response = self.get_product_details(entities)
        elif intent == "employee_details":
            response = self.get_employee_details(entities)
        elif intent == "service_details":  # Add this condition for the "service_details" intent
             response = self.get_service_details(entities)
        else:
            response = self.get_answer_from_chatgpt(user_text)
        #dispatcher.utter_message('Response (custom_action): ' + str(response)) # for testing are they custom actions return calls
        dispatcher.utter_message(str(response))
        return []

    def get_customer_details(self, entities: List[Dict[Text, Any]]) -> Text:
         if not entities:
           return "No entity provided."
         credentials = ServiceAccountCredentials.from_json_keyfile_name(os.getenv("GOOGLE_AUTH_FILE"), ['https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive'])
         client = gspread.authorize(credentials)

         sheet = client.open('Ziggy_DB_Sheet')  
         customer_data = sheet.worksheet('Customer_Details')
         customer_records = customer_data.get_all_records()

         entity_value = entities[0]['value']  
         customer = next((record for record in customer_records if record['Customer ID'] == entity_value
                     or record['First Name'] == entity_value
                     or record['Email'] == entity_value
                     or record['Last Name'] == entity_value), None)
         if customer:
                answer = f"{customer['First Name']} {customer['Last Name']} is the customer with customer ID {customer['Customer ID']}. "
                answer += f"His email is {customer['Email']} and his contact number is {customer['Phone Number']}. "
                answer += f"He lives at {customer['Address']}."
         else:
            answer = "Sorry !! Customer not found."
         return answer


    def get_product_details(self, entities: List[Dict[Text, Any]]) -> Text:
        credentials = ServiceAccountCredentials.from_json_keyfile_name(os.getenv("GOOGLE_AUTH_FILE"), ['https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive'])
        client = gspread.authorize(credentials)

        sheet = client.open('Ziggy_DB_Sheet')  
        product_data = sheet.worksheet('Product_Details')
        product_records = product_data.get_all_records()

        entity_value = entities[0]['value']  
        product = next((record for record in product_records if record['Product ID'] == entity_value), None)
        
        if not product:
            product = next((record for record in product_records if record['Product Name'] == entity_value), None)

        if product:
            answer = f"Product ID: {product['Product ID']}\n"
            answer += f"Product Name: {product['Product Name']}\n"
            answer += f"Category: {product['Category']}\n"
            answer += f"Price: {product['Price']}\n"
            answer += f"Description: {product['Description']}\n"
            answer += f"In Stock: {product['In Stock']}"
        else:
            answer = "Sorry !! Product not found."
        
        return answer
    
    def get_employee_details(self, entities: List[Dict[Text, Any]]) -> Text:
        credentials = ServiceAccountCredentials.from_json_keyfile_name(os.getenv("GOOGLE_AUTH_FILE"), ['https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive'])
        client = gspread.authorize(credentials)

        sheet = client.open('Ziggy_DB_Sheet')  
        employee_data = sheet.worksheet('Employee_Details')
        employee_records = employee_data.get_all_records()
    
        entity_value = entities[0]['value']  
        employee = next((record for record in employee_records if record['Employee ID'] == entity_value
                     or record['First Name'] == entity_value), None)
        if employee:
            gender_pronoun = "he" if employee['Gender'].lower() == "male" else "she"
            answer = f" {employee['First Name']} {employee['Last Name']} is born on {employee['Date of Birth']} and employee here with employee ID {employee['Employee ID']}.\n"
            answer += f"{gender_pronoun.capitalize()} works under the {employee['Department']} department as a {employee['Position']} with a pay of {employee['Salary']}.\n"
            answer += f"His email is {employee['Email']} and his contact number is {employee['Phone Number']}. He lives at {employee['Address']}."
        else:
            answer = "Sorry !! Employee not found."
        
        return answer

    def get_service_details(self, entities: List[Dict[Text, Any]]) -> Text:
        credentials = ServiceAccountCredentials.from_json_keyfile_name(os.getenv("GOOGLE_AUTH_FILE"), ['https://www.googleapis.com/auth/spreadsheets','https://www.googleapis.com/auth/drive'])
        client = gspread.authorize(credentials)

        sheet = client.open('Ziggy_DB_Sheet')  
        service_data = sheet.worksheet('Service_Details')
        service_records = service_data.get_all_records()
    
        entity_value = entities[0]['value']  
        service = next((record for record in service_records if record['Service ID'] == entity_value
                or record['Service Name'] == entity_value), None)
    
        if service:
            availability = "available" if service['Availability'].lower() == "yes" else "unavailable"
            answer = f"For service {service['Service Name']} having service ID {service['Service ID']} of {service['Category']} for a price of {service['Price']}, it is {availability}.\n"
            answer += f"{service['Description']}"
        else:
            answer = "Service not found."

        return answer


    def get_answer_from_chatgpt(self, user_text: Text) -> Text:
        openai.api_key = os.getenv("GPT_API_KEY")
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=user_text,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        ).choices[0].text.strip()
        return response
