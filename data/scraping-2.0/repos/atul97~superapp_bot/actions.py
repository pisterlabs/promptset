from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import AllSlotsReset, SlotSet


import requests
import psycopg2
import openai

api_key = "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"
openai.api_key = api_key

class ActionSendOTP(Action):

    def name(self) -> Text:
        return "action_sendotp"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        authentic_user = tracker.get_slot("authentic_user")
        otp = tracker.get_slot("otp")


        conn = None
        try:
            mobile = tracker.get_slot("mobile")
            print(mobile)
            conn = psycopg2.connect(database ="sapp", user = "postgres",
                        password = "123456", host = "localhost", 
                        port = "5432")
            print("Connection Successful to PostgreSQL")

            cur = conn.cursor()
            
            query = f"""select * from customer where phone_number = '{mobile}';"""
            print(query)
            cur.execute(query)
            rows = cur.fetchall()


            
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            
            print(error)
        finally:
            if conn is not None:
                conn.close()
                print('Database connection closed.')
        if len(rows)>0:
            # dispatcher.utter_message(text=f"No details found for user with contact: {mobile} ")
            print("User Exist")
            authentic_user = "True"
            otp = "1234"




        if authentic_user !="True":
            try:
                
                number = tracker.get_slot("mobile")
                country_code = tracker.get_slot("country_code")
                url = "https://verificationapi-v1.sinch.com/verification/v1/verifications"
                payload=            {
                        "identity": {
                            "type": "number",
                            "endpoint": f"{country_code}{number}"},
                            "method": "sms"
                            }
                payload = str(payload)
                # payload="{\n  \"identity\": {\n  \"type\": \"number\",\n  \"endpoint\": \"+919873147995\"\n  },\n  \"method\": \"sms\"\n}"
                headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Basic MDM2YjIxM2UtMDdlOS00ZTNiLWJlOTAtMmI1NDRjMGI1OGU1Ok1jL1Z1MEFkbWtPQXUwc0dlNEo4blE9PQ=='
                }
                response = requests.request("POST", url, headers=headers, data=payload)
                dispatcher.utter_message(text=f"OTP sent to {country_code}{number}")
            except:
                print(response.json())
                dispatcher.utter_message(text="Something went wrong")
            

        return [SlotSet('authentic_user',authentic_user),SlotSet('otp',otp)]

class ActionVerifyOTP(Action):

    def name(self) -> Text:
        return "action_verify_otp"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:


        authentic_user = tracker.get_slot("authentic_user") 
        print(f"Auth_user:{authentic_user}")
        if authentic_user !="True":
            try:
                otp = tracker.get_slot("otp")
                number = tracker.get_slot("mobile")
                country_code = tracker.get_slot("country_code")
                url = f"https://verificationapi-v1.sinch.com/verification/v1/verifications/number/{country_code}{number}"
                # payload="{ \"method\": \"sms\", \"sms\":{ \"code\": \"4163\" }}"
                payload={ 
                "method": "sms", 
                "sms":{ 
                    "code": f"{otp}" 
                    }
                    }
                payload = str(payload)

                headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Basic MDM2YjIxM2UtMDdlOS00ZTNiLWJlOTAtMmI1NDRjMGI1OGU1Ok1jL1Z1MEFkbWtPQXUwc0dlNEo4blE9PQ=='
                }
                response = requests.request("PUT", url, headers=headers, data=payload)

                print(response.json())
                response_json = response.json()
                dispatcher.utter_message(text=f"Verification Status:{response_json['status']}")
                if response_json['status'] == 'SUCCESSFUL':
                    return [SlotSet('authentic_user',"True")]
                else:
                    return [AllSlotsReset()]
            except:
                dispatcher.utter_message(text="Something went wrong")

        return []


class ActionCheckCustomer(Action):

    def name(self) -> Text:
        return "action_check_customer"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:      
        

        authentic_user = tracker.get_slot("authentic_user") 

        if authentic_user =="True":
            conn = None
            try:
                print("-----------")
                mobile = tracker.get_slot("mobile")
                conn = psycopg2.connect(database ="sapp", user = "postgres",
                            password = "123456", host = "localhost", 
                            port = "5432")
                print("Connection Successful to PostgreSQL ActionCheckCustomer")

                cur = conn.cursor()
                
                query = f"""select * from customer where phone_number = '{mobile}' and flag_exist = 1;"""
                cur.execute(query)
                rows = cur.fetchall()


                
                cur.close()
            except (Exception, psycopg2.DatabaseError) as error:
                
                print(error)
            finally:
                if conn is not None:
                    conn.close()
                    print('Database connection closed.')
            if len(rows)==0:
                # dispatcher.utter_message(text=f"No details found for user with contact: {mobile} ")

                try:
                    row2 = 0
                    print("-----------")
                    mobile = tracker.get_slot("mobile")
                    conn = psycopg2.connect(database ="sapp", user = "postgres",
                                password = "123456", host = "localhost", 
                                port = "5432")
                    print("Connection Successful to PostgreSQL if user not existed")

                    cur = conn.cursor()
                    
                    query = f"""select * from customer where phone_number = '{mobile}';"""
                    cur.execute(query)
                    row2 = cur.fetchall()
                    
                    cur.close()
                except (Exception, psycopg2.DatabaseError) as error:
                    
                    print(error)
                finally:
                    if conn is not None:
                        conn.close()
                        print('Database connection closed.')

                if len(row2)==0:
                    dispatcher.utter_message(text=f"No details found for user with contact: {mobile}")

                    buttons = []
                    buttons.append({"title": "Add new customer" , "payload": "add_customer"})
                    buttons.append({"title": "Explore More" , "payload": "explore_more"})
                    dispatcher.utter_message(text="Want to add customer",buttons=buttons)

                else:
                    buttons = []
                    buttons.append({"title": "Activate customer subscription" , "payload": "activate_customer"})
                    buttons.append({"title": "Explore More" , "payload": "explore_more"})
                    dispatcher.utter_message(text="Want to activate customer subscription",buttons=buttons)
            else:
                # dispatcher.utter_message(text="User existed")
                dispatcher.utter_message(template=f"utter_main_menu")
        else:
            dispatcher.utter_message(text=f"Please authenticate")

        return []



class ActionActivateCustomer(Action):

    def name(self) -> Text:
        return "action_activate_customer"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        authentic_user = tracker.get_slot("authentic_user") 

        if authentic_user =="True":
            conn = None
            try:
                print("-----------")
                mobile = tracker.get_slot("mobile")

                conn = psycopg2.connect(database ="sapp", user = "postgres",
                            password = "123456", host = "localhost", 
                            port = "5432")
                print("Connection Successful to PostgreSQL")

                cur = conn.cursor()
                
                query = f"""update customer set flag_exist = 1 where phone_number = '{mobile}';"""
                cur.execute(query)
                conn.commit()

                
                cur.close()
            except (Exception, psycopg2.DatabaseError) as error:
                
                print(error)
            finally:
                if conn is not None:
                    conn.close()
                    print('Database connection closed.')
        else:
            dispatcher.utter_message(text=f"Please authenticate")
        return [SlotSet('order_type',None)]


class ActionDeactivateCustomer(Action):

    def name(self) -> Text:
        return "action_deactivate_customer"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        authentic_user = tracker.get_slot("authentic_user") 

        if authentic_user =="True":
            conn = None
            try:
                print("-----------")
                mobile = tracker.get_slot("mobile")

                conn = psycopg2.connect(database ="sapp", user = "postgres",
                            password = "123456", host = "localhost", 
                            port = "5432")
                print("Connection Successful to PostgreSQL")

                cur = conn.cursor()
                
                query = f"""update customer set flag_exist = 0 where phone_number = '{mobile}';"""
                cur.execute(query)
                conn.commit()

                
                cur.close()
                dispatcher.utter_message(text=f"Customer subscription has been deactivated")
            except (Exception, psycopg2.DatabaseError) as error:
                
                print(error)
            finally:
                if conn is not None:
                    conn.close()
                    print('Database connection closed.')
        else:
            dispatcher.utter_message(text=f"Please authenticate")
        return [SlotSet('order_type',None)]


class ActionExistingOrder(Action):

    def name(self) -> Text:
        return "action_existing_order"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        authentic_user = tracker.get_slot("authentic_user") 

        if authentic_user =="True":
            conn = None
            try:
                # print("-----------")
                mobile = tracker.get_slot("mobile")
                order_type = tracker.get_slot("order_type")
                order_type = order_type.lower()
                conn = psycopg2.connect(database ="sapp", user = "postgres",
                            password = "123456", host = "localhost", 
                            port = "5432")
                # print("Connection Successful to PostgreSQL")

                cur = conn.cursor()
                
                query = f"""select * from orders where phone_number = '{mobile}' and order_type = '{order_type}';"""
                cur.execute(query)
                rows = cur.fetchall()
                for i in rows :
                    dispatcher.utter_message(text=f"Data: {i[2]} , Status: {i[3]}, Amount: {i[4]}")

                
                cur.close()
            except (Exception, psycopg2.DatabaseError) as error:
                
                print(error)
            finally:
                if conn is not None:
                    conn.close()
                    # print('Database connection closed.')
            if len(rows)==0:
                dispatcher.utter_message(text=f"No details found for user with contact: {mobile}  for {order_type} order")
        else:
            dispatcher.utter_message(text=f"Please authenticate")
        return [SlotSet('order_type',None)]


class ActionAddCustomer(Action):

    def name(self) -> Text:
        return "action_add_customer"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:


        authentic_user = tracker.get_slot("authentic_user") 

        if authentic_user =="True":        
            conn = None
            try:
                # print("-----------")
                mobile = tracker.get_slot("mobile")
                customer_name = tracker.get_slot("customer_name")
                country_code = tracker.get_slot("country_code")
                conn = psycopg2.connect(database ="sapp", user = "postgres",
                            password = "123456", host = "localhost", 
                            port = "5432")
                # print("Connection Successful to PostgreSQL")

                cur = conn.cursor()
                
                query = f"""insert into customer values('{customer_name}',{mobile},'{country_code}');"""
                cur.execute(query)

                conn.commit()


                
                cur.close()
                dispatcher.utter_message(template=f"utter_main_menu")
            except (Exception, psycopg2.DatabaseError) as error:
                
                print(error)
            finally:
                if conn is not None:
                    conn.close()
                    # print('Database connection closed.')
        else:
            dispatcher.utter_message(text=f"Please authenticate")

        
        return [SlotSet('order_type',None)]


class ActionEnquiryExistingOrder(Action):

    def name(self) -> Text:
        return "action_enquiry_existing_order"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # authentic_user = tracker.get_slot("authentic_user") 
        authentic_user = "True"

        if authentic_user =="True":
            conn = None
            try:
                # print("-----------")
                # mobile = tracker.get_slot("mobile")
                # order_type = tracker.get_slot("order_type")
                # order_type = order_type.lower()
                mobile = "9953635285"
                order_type = "hyperlocal"
                conn = psycopg2.connect(database ="sapp", user = "postgres",
                            password = "123456", host = "localhost", 
                            port = "5432")
                # print("Connection Successful to PostgreSQL")

                cur = conn.cursor()
                
                query = f"""select * from orders where phone_number = '{mobile}' and order_type = '{order_type}';"""
                cur.execute(query)
                rows = cur.fetchall()
                for i in rows :
                    dispatcher.utter_message(text=f"Data: {i[2]} , Status: {i[3]}, Amount: {i[4]}")

                
                cur.close()
            except (Exception, psycopg2.DatabaseError) as error:
                
                print(error)
            finally:
                if conn is not None:
                    conn.close()
                    # print('Database connection closed.')
            if len(rows)==0:
                dispatcher.utter_message(text=f"No details found for user with contact: {mobile}  for {order_type} order")
        else:
            dispatcher.utter_message(text=f"Please authenticate")
        return [SlotSet('order_type',None)]


class ActionResourcesList(Action):

    def name(self) -> Text:
        return "action_resources_list"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        test_carousel = {
            "type": "template",
            "payload": {
                "template_type": "generic",
                "elements": [{
                    "title": "Innovate Youself",
                    "subtitle": "Get It, Make it.",
                    "image_url": "static/test.jpg",
                    "buttons": [{
                        "title": "Innovate Yourself",
                        "url": "https://www.innovationyourself.com/",
                        # "https://yt3.ggpht.com/ytc/AAUvwnhZwcqP89SH71KugPDfltbcpBajoPpxihN7aPGOmzE=s900-c-k-c0x00ffffff-no-rj",
                        "type": "web_url"
                    },
                        {
                            "title": "Innovate Yourself",
                            "type": "postback",
                            "payload": "/greet"
                        }
                    ]
                },
                    {
                        "title": "RASA CHATBOT",
                        "subtitle": "Conversational AI",
                        "image_url": "static/rasa.png",
                        "buttons": [{
                            "title": "Rasa",
                            "url": "https://www.rasa.com",
                            "type": "web_url"
                        },
                            {
                                "title": "Rasa Chatbot",
                                "type": "postback",
                                "payload": "/greet"
                            }
                        ]
                    }
                ]
            }
        }
        covid_resources = {
            "type": "template",
            "payload": {
                "template_type": "generic",
                "elements": [{
                    "title": "MBMC",
                    "subtitle": "FIND BED, SAVE LIFE.",
                    "image_url": "static/hospital-beds-application.jpg",
                    "buttons": [{
                        "title": "Hospital Beds Availability",
                        "url": "https://www.covidbedmbmc.in/",
                        "type": "web_url"
                    },
                        {
                            "title": "MBMC",
                            "type": "postback",
                            "payload": "/affirm"
                        }
                    ]
                },
                    {
                        "title": "COVID.ARMY",
                        "subtitle": "OUR NATION, SAVE NATION.",
                        "image_url": "static/oxygen-cylinder-55-cft-500x554-500x500.jpg",
                        "buttons": [{
                            "title": "COVID ARMY",
                            "url": "https://covid.army/",
                            "type": "web_url"
                        },
                            {
                                "title": "COVID ARMY",
                                "type": "postback",
                                "payload": "/deny"
                            }
                        ]
                    },
                    {
                        "title": "Innovate Youself",
                        "subtitle": "Get It, Make it.",
                        "image_url": "static/test.jpg",
                        "buttons": [{
                            "title": "Innovate Yourself",
                            "url": "https://www.innovationyourself.com/",
                            # "https://yt3.ggpht.com/ytc/AAUvwnhZwcqP89SH71KugPDfltbcpBajoPpxihN7aPGOmzE=s900-c-k-c0x00ffffff-no-rj",
                            "type": "web_url"
                        },
                            {
                                "title": "Innovate Yourself",
                                "type": "postback",
                                "payload": "/greet"
                            }
                        ]
                    },
                    {
                        "title": "RASA CHATBOT",
                        "subtitle": "Conversational AI",
                        "image_url": "static/rasa.png",
                        "buttons": [{
                            "title": "Rasa",
                            "url": "https://www.rasa.com",
                            "type": "web_url"
                        },
                            {
                                "title": "Rasa Chatbot",
                                "type": "postback",
                                "payload": "/greet"
                            }
                        ]
                    }
                ]
            }
        }

        dispatcher.utter_message(attachment=test_carousel)
        return []





# class ActionWriteToDB(Action):

#     def name(self) -> Text:
#         return "action_Write_to_DB"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

#         authentic_user = "True"

#         if authentic_user =="True":
#             conn = None
#             try:
#                 print("-----------")

#                 mobile = "9953635285"
#                 order_type = "hyperlocal"
#                 conn = psycopg2.connect(database ="sapp", user = "postgres",
#                             password = "123456", host = "localhost", 
#                             port = "5432")
#                 print("Connection Successful to PostgreSQL")

#                 cur = conn.cursor()
                
#                 query = f"""select * from orders where phone_number = '{mobile}' and order_type = '{order_type}';"""
#                 cur.execute(query)
#                 rows = cur.fetchall()
#                 for i in rows :
#                     dispatcher.utter_message(text=f"Data: {i[2]} , Status: {i[3]}, Amount: {i[4]}")

                
#                 cur.close()
#             except (Exception, psycopg2.DatabaseError) as error:
                
#                 print(error)
#             finally:
#                 if conn is not None:
#                     conn.close()
#                     print('Database connection closed.')
#             if len(rows)==0:
#                 dispatcher.utter_message(text=f"No details found for user with contact: {mobile}  for {order_type} order")
#         else:
#             dispatcher.utter_message(text=f"Please authenticate")
#         return [SlotSet('order_type',None)]



# class ActionRemoveCustomer(Action):

#     def name(self) -> Text:
#         return "action_remove_customer"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

#         authentic_user = "True"

#         if authentic_user =="True":
#             conn = None
#             try:
#                 print("-----------")

#                 mobile = "9953635285"
#                 order_type = "hyperlocal"
#                 conn = psycopg2.connect(database ="sapp", user = "postgres",
#                             password = "123456", host = "localhost", 
#                             port = "5432")
#                 print("Connection Successful to PostgreSQL")

#                 cur = conn.cursor()
                
#                 query = f"""select * from orders where phone_number = '{mobile}' and order_type = '{order_type}';"""
#                 cur.execute(query)
#                 rows = cur.fetchall()
#                 for i in rows :
#                     dispatcher.utter_message(text=f"Data: {i[2]} , Status: {i[3]}, Amount: {i[4]}")

                
#                 cur.close()
#             except (Exception, psycopg2.DatabaseError) as error:
                
#                 print(error)
#             finally:
#                 if conn is not None:
#                     conn.close()
#                     print('Database connection closed.')
#             if len(rows)==0:
#                 dispatcher.utter_message(text=f"No details found for user with contact: {mobile}  for {order_type} order")
#         else:
#             dispatcher.utter_message(text=f"Please authenticate")
#         return [SlotSet('order_type',None)]





# class ActionRemoveCustomer(Action):

#     def name(self) -> Text:
#         return "action_remove_customer"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

#         authentic_user = "True"

#         if authentic_user =="True":
#             conn = None
#             try:
#                 print("-----------")

#                 mobile = "9953635285"
#                 order_type = "hyperlocal"
#                 conn = psycopg2.connect(database ="sapp", user = "postgres",
#                             password = "123456", host = "localhost", 
#                             port = "5432")
#                 print("Connection Successful to PostgreSQL")

#                 cur = conn.cursor()
                
#                 query = f"""select * from orders where phone_number = '{mobile}' and order_type = '{order_type}';"""
#                 cur.execute(query)
#                 rows = cur.fetchall()
#                 for i in rows :
#                     dispatcher.utter_message(text=f"Data: {i[2]} , Status: {i[3]}, Amount: {i[4]}")

                
#                 cur.close()
#             except (Exception, psycopg2.DatabaseError) as error:
                
#                 print(error)
#             finally:
#                 if conn is not None:
#                     conn.close()
#                     print('Database connection closed.')
#             if len(rows)==0:
#                 dispatcher.utter_message(text=f"No details found for user with contact: {mobile}  for {order_type} order")
#         else:
#             dispatcher.utter_message(text=f"Please authenticate")
#         return [SlotSet('order_type',None)]



class ActionDefaultFallback(Action):
    """Executes the fallback action and goes back to the previous state
    of the dialogue"""

    def name(self) -> Text:
        return "action_default_fallback"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(template="utter_please_rephrase")

        # Revert user message which led to fallback.
        return []

def send_message(message_log):
    # Use OpenAI's ChatCompletion API to get the chatbot's response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # The name of the OpenAI chatbot model to use
        messages=message_log,   # The conversation history up to this point, as a list of dictionaries
        max_tokens=3800,        # The maximum number of tokens (words or subwords) in the generated response
        stop=None,              # The stopping sequence for the generated response, if any (not used here)
        temperature=0.7,        # The "creativity" of the generated response (higher temperature = more creative)
    )
    return response.choices[0].message.content

class ActionGPTFallback(Action):
    """Executes the fallback action and goes back to the previous state
    of the dialogue"""

    def name(self) -> Text:
        return "action_gpt_fallback"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        
        msg = tracker.latest_message['text']

        message_log = [{"role": "system", "content": "You are a helpful assistant."},
               {"role": "user", "content": msg},
        ]

        
        response = send_message(message_log)
        print(response)

        dispatcher.utter_message(text=response)

        # Revert user message which led to fallback.
        return []