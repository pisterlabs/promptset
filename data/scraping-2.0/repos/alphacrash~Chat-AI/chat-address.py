# RUN: python chat.py
import json
import openai
import requests
import prompts

# Values
key_file = "values/key.json"
urls_file = "values/urls.json"

# Key
with open(key_file, "r") as file:
    key_data = json.load(file)
with open(urls_file, "r") as file:
    urls_data = json.load(file)

openai.api_key = key_data["key"]

# API Endpoints
get_url = urls_data["getAddresses"]
create_url = urls_data["createAddress"]
update_url = urls_data["updateAddress"]
delete_url = urls_data["deleteAddress"]

# Main
class AddressChat:
    CONVERSATION_AGREE_PROMPT = """OK"""
    CONVERSATION_START_PROMPT = """Great! Start the Conversation."""
    CONVERSATION_PROMPT = """You are conversation assistant that manages addresses of consumers. Your task is to follow the conversation flow to assist the consumer.
    
    ###
    Conversation Flow:
    1. Greet the consumer
    2. Check if they need any assistance.
    3. Answer their requests
    4. Greet the consumer and end the conversation by responding '[END_OF_CONVERSATION]'
    ###

    """

    INTENT_DETECTION_SETUP_PROMPT = """Your task is to classify the consumer's intent from the below `Conversation` into following `Intent Categories`. Response should follow the `Output Format`.

    Conversation:
    {conversation}

    Intent Categories:
    GREETING: consumer is greeting the chatbot.
    GET_ADDRESSES: consumer's request to view his saved addresses.
    CREATE_ADDRESS: consumer's request to create a new address.
    UPDATE_ADDRESS: consumer's request to update his saved address.
    DELETE_ADDRESS: consumer's request to remove/delete his saved address. 
    OUT_OF_CONTEXT: consumer's query is irrelevant and cannot be classified in the above intents.

    Output Format: <PREDICTED_INTENT>
    """

    ENQUIRY_DETAILS_PROMPT = """Your task is to extract the following `Entities` from the below `Conversation` between an assistant and a consumer. Response should follow the `Output Format`. If some entities are missing provide NULL in the `Output Format`.

    Conversation:
    {conversation}

    Entities:
    CONSUMER_ID: This is the id of the consumer.
    STREET: This is the street name of the address.
    CITY: This is the city name of the address.
    STATE: This is the state name of the address.
    ZIP_CODE: This is the zip code of the address.
    ADDRESS_TYPE: This is the type of address. It can be either 'Home' or 'Mail'.

    Output Format: {{'CONSUMER_ID': <Consumer ID in strings>, 'STREET': <Street name in strings>, 'CITY': <City name in strings>, 'STATE': <State name in strings>, 'ZIP_CODE': <Zip code in strings>, 'ADDRESS_TYPE': <Address type in strings>}}
    """

    def intent_detection(self, conversation):
        chat_ml = [
                    {"role": "user", "content": self.INTENT_DETECTION_SETUP_PROMPT.format(conversation=conversation)}
                  ]
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_ml,
        temperature=0)
        
        return response['choices'][0]['message']['content'].strip(" \n'")
    

    def enquiry_details(self, conversation):
        chat_ml = [
            {"role": "user", "content": self.ENQUIRY_DETAILS_PROMPT.format(conversation=conversation)}
        ]
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat_ml,
        temperature=0)
        
        return response['choices'][0]['message']['content'].strip(" \n")

    def conversation_chat(self):
        conversation = ""
        end_flag = False

        chatml_messages = [
            {"role": "user", "content": self.CONVERSATION_PROMPT},
            {"role": "assistant", "content": self.CONVERSATION_AGREE_PROMPT},
            {"role": "user", "content": self.CONVERSATION_START_PROMPT}
        ]

        while True:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=chatml_messages
            )

            agent_response = response['choices'][0]['message']['content'].strip(" \n")

            if "END_OF_CONVERSATION" in agent_response:
                print("Assistant: Thank you for connecting with us. Have a nice day!")
                break
            elif end_flag==True:
                print("Assistant: {}".format(agent_response))
                print("Assistant: Thank you for connecting with us. Have a nice day!")
                break

            print("Assistant: {}".format(agent_response))
            chatml_messages.append({"role": "assistant", "content": agent_response})
            conversation += "Assistant: {}\n".format(agent_response)

            consumer_response = input("Consumer: ")
            if consumer_response == "/end":
                break
            chatml_messages.append({"role": "user", "content": consumer_response})
            conversation += "Consumer: {}\n".format(consumer_response)

             # Classify the intent
            intent = self.intent_detection(conversation)
            # print("Intent: {}".format(intent))

            if 'OUT_OF_CONTEXT' in intent:
                chatml_messages.append({"role": "user", "content": "Politely say to consumer to stay on the topic not to diverge."})
            elif 'GREETING' in intent:
                chatml_messages.append({"role": "user", "content": "Greet the consumer and ask how you can help them."})
            elif 'GET_ADDRESSES' in intent:
                entities = self.enquiry_details(conversation)
                entities = entities.split(",")
                consumer_id = entities[0].split(":")[-1].strip(" '}{")

                if consumer_id.upper() == "NULL":
                    chatml_messages.append({"role": "user", "content": "Ask the consumer for their ID"})
                else:
                    response = requests.get(get_url.replace("consumerId", consumer_id))
                    resp_json = response.json()

                    if response.status_code == 200:
                        chatml_messages.append({"role": "user", "content": "Provide the details in natural language, don't display in json format to the consumer and mention no addresses if not found:\n{}".format(str(resp_json))})
                        end_flag = True
                    else:
                        chatml_messages.append({"role": "user", "content": "Some invalid data is provided. Provide the details to the consumer as depicted in json in natural language, don't display in json format\n{}".format(str(resp_json))})
                        end_flag = True
            elif 'CREATE_ADDRESS' in intent:
                entities = self.enquiry_details(conversation)
                entities = entities.split(",")
                consumer_id = entities[0].split(":")[-1].strip(" '}{")
                street = entities[1].split(":")[-1].strip(" '}{")
                city = entities[2].split(":")[-1].strip(" '}{")
                state = entities[3].split(":")[-1].strip(" '}{")
                zip_code = entities[4].split(":")[-1].strip(" '}{")
                address_type = entities[5].split(":")[-1].strip(" '}{")

                if consumer_id.upper() == "NULL":
                    chatml_messages.append({"role": "user", "content": "Ask the consumer for their ID"})
                elif street.upper() == "NULL":
                    chatml_messages.append({"role": "user", "content": "Ask the consumer for street name"})
                elif city.upper() == "NULL":
                    chatml_messages.append({"role": "user", "content": "Ask the consumer for city name"})
                elif state.upper() == "NULL":
                    chatml_messages.append({"role": "user", "content": "Ask the consumer for state name"})
                elif zip_code.upper() == "NULL":
                    chatml_messages.append({"role": "user", "content": "Ask the consumer for zip code"})
                elif address_type.upper() == "NULL" or address_type.upper() not in ["HOME", "MAIL"]:
                    chatml_messages.append({"role": "user", "content": "Ask the consumer for address type. It can be either 'Home' or 'Mail'"})
                else:
                    data = {
                        "consumerId": consumer_id,
                        "street": street,
                        "city": city,
                        "state": state,
                        "zipCode": zip_code,
                        "addressType": address_type
                    }
                    response = requests.post(create_url.replace("consumerId", consumer_id), json=data)
                    resp_json = response.json()

                    if response.status_code == 200:
                        response = requests.get(get_url.replace("consumerId", consumer_id))
                        resp_json = response.json()
                        chatml_messages.append({"role": "user", "content": "Inform that address is created and display created details in natural language, not in json:\n{}".format(str(resp_json))})
                        end_flag = True
                    else:
                        chatml_messages.append({"role": "user", "content": "Some invalid data is provided by the consumer. Provide the details to the consumer in natural language, don't display json:\n{}".format(str(resp_json))})
                        end_flag = True
            elif 'UPDATE_ADDRESS' in intent:
                entities = self.enquiry_details(conversation)
                entities = entities.split(",")
                consumer_id = entities[0].split(":")[-1].strip(" '}{")
                street = entities[1].split(":")[-1].strip(" '}{")
                city = entities[2].split(":")[-1].strip(" '}{")
                state = entities[3].split(":")[-1].strip(" '}{")
                zip_code = entities[4].split(":")[-1].strip(" '}{")
                address_type = entities[5].split(":")[-1].strip(" '}{")

                if consumer_id.upper() == "NULL":
                    chatml_messages.append({"role": "user", "content": "Ask the consumer for their ID"})
                elif street.upper() == "NULL":
                    chatml_messages.append({"role": "user", "content": "Ask the consumer for street name"})
                elif city.upper() == "NULL":
                    chatml_messages.append({"role": "user", "content": "Ask the consumer for city name"})
                elif state.upper() == "NULL":
                    chatml_messages.append({"role": "user", "content": "Ask the consumer for state name"})
                elif zip_code.upper() == "NULL":
                    chatml_messages.append({"role": "user", "content": "Ask the consumer for zip code"})
                elif address_type.upper() == "NULL" or address_type.upper() not in ["HOME", "MAIL"]:
                    chatml_messages.append({"role": "user", "content": "Ask the consumer for address type. It can be either 'Home' or 'Mail'"})
                else:
                    data = {
                        "consumerId": consumer_id,
                        "street": street,
                        "city": city,
                        "state": state,
                        "zipCode": zip_code,
                        "addressType": address_type
                    }
                    response = requests.put(update_url.replace("consumerId", consumer_id), json=data)

                    if response.status_code == 200:
                        response = requests.get(get_url.replace("consumerId", consumer_id))
                        resp_json = response.json()
                        chatml_messages.append({"role": "user", "content": "Inform that address is updated and display updated details in natural language, not in json:\n{}".format(str(resp_json))})
                        end_flag = True
                    else:
                        chatml_messages.append({"role": "user", "content": "Some invalid data is provided by the consumer. Provide the details to the consumer in natural language, don't display json:\n{}".format(str(resp_json))})
                        end_flag = True
            elif 'DELETE_ADDRESS' in intent:
                entities = self.enquiry_details(conversation)
                entities = entities.split(",")
                consumer_id = entities[0].split(":")[-1].strip(" '}{")
                address_type = entities[5].split(":")[-1].strip(" '}{")

                if consumer_id.upper() == "NULL":
                    chatml_messages.append({"role": "user", "content": "Ask the consumer for their ID"})
                elif address_type.upper() == "NULL" or address_type.upper() not in ["HOME", "MAIL"]:
                    chatml_messages.append({"role": "user", "content": "Ask the consumer for address type. It can be either 'Home' or 'Mail'"})
                else:
                    response = requests.delete(delete_url.replace("consumerId", consumer_id).replace("addressType", address_type))

                    if response.status_code == 200:
                        chatml_messages.append({"role": "user", "content": "Inform that book is deleted"})
                        end_flag = True
                    else:
                        chatml_messages.append({"role": "user", "content": "Some invalid data is provided by the consumer."})
                        end_flag = True


if __name__ == "__main__":
    AC = AddressChat()
    AC.conversation_chat()