from openai import OpenAI
import time
import yfinance as yf
from dotenv import load_dotenv
import os
import json
import requests

# Load environment variables from a .env file
load_dotenv()
base_api_url = os.getenv("base_api_url")

# Functions to interact with a real estate API
def get_real_estate_agent_listed_properties_by_id(funda_id: int):
    response = requests.get(base_api_url + "/GetRealEstateBrokerById/" + str(funda_id))
    return response.json()

def get_top_real_estate_brokers_with_the_most_amount_of_homes():
    response = requests.get(base_api_url + "/GetTopRealEstateBrokersWithTheMostAmountOfHomes")
    return response.json()

def get_top_real_estate_brokers_with_the_most_amount_of_homes_with_garden():
    response = requests.get(base_api_url + "/GetTopRealEstateBrokersWithTheMostAmountOfHomesWithGarden")
    return response.json()

def get_top_real_estate_brokers_with_the_most_amount_of_homes_with_balcony_or_terrace():
    response = requests.get(base_api_url + "/GetTopRealEstateBrokersWithTheMostAmountOfHomesWithBalconyOrTerrace")
    return response.json()

def get_top_real_estate_brokers_with_the_most_amount_of_homes_with_garage():
    response = requests.get(base_api_url + "/GetTopRealEstateBrokersWithTheMostAmountOfHomesWithGarage")
    return response.json()

# Class to manage interactions with the OpenAI assistant
class AssistantManager:
    def __init__(self, api_key, model = "gpt-4-1106-preview"):
        self.client = OpenAI(api_key = api_key)
        self.model = model
        self.assistant = None
        self.thread = None
        self.run = None

    # Create a new assistant with specified instructions and tools
    def create_assistant(self, name, instructions, tools):
        self.assistant = self.client.beta.assistants.create(
            name = name,
            instructions = instructions,
            tools = tools,
            model = self.model
        )

     # Create a specific assistant for interacting with real estate data
    def create_funda_assistant(self):
        self.create_assistant(
            name = "Funda Real Estate Brokers Assistant 2",
            instructions = "You are a personal Real Estate Brokers Assistant",
            tools = [{
                "type": "function",
                "function": {
                    "name": "get_top_real_estate_brokers_with_the_most_amount_of_homes",
                    "description": "Retrive top 15 real estate brokers with the most amount of homes",
                   "parameters": {
                        "type": "object",
                        "properties": {                        
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_top_real_estate_brokers_with_the_most_amount_of_homes_with_garden",
                    "description": "Retrieve top 15 real estate brokers with the most amount of homes with garden",
                    "parameters": {
                        "type": "object",
                        "properties": {                        
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_top_real_estate_brokers_with_the_most_amount_of_homes_with_balcony_or_terrace",
                    "description": "Retrieve top 15 real estate brokers with the most amount of homes with balcony or terrace",
                    "parameters": {
                        "type": "object",
                        "properties": {                        
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_top_real_estate_brokers_with_the_most_amount_of_homes_with_garage",
                    "description": "Retrieve top 15 real estate brokers with the most amount of homes with garage",
                    "parameters": {
                        "type": "object",
                        "properties": {                        
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_real_estate_agent_listed_properties_by_id",
                    "description": "Retrieve all listed properties by a real estate agent",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "funda_id": {
                                "type": "integer",
                                "description": "The FundaId of the real estate agent"
                            }
                        },
                        "required": ["FundaId"]
                    }
                }
            }]
        )
    
    # List all available assistants
    def list_assistants(self):
        return self.client.beta.assistants.list()
    
    # Select a specific assistant by ID
    def select_assistant(self, assistant_id):
        self.assistant = self.client.beta.assistants.retrieve(assistant_id)

    # Create a new conversation thread
    def create_thread(self):
        self.thread = self.client.beta.threads.create()

    # Add a message to the conversation thread
    def add_message_to_thread(self, role, content):
        self.client.beta.threads.messages.create(
            thread_id = self.thread.id,
            role = role,
            content = content
        )

    # Run the assistant on the created thread with given instructions
    def run_assistant(self, instructions):
        self.run = self.client.beta.threads.runs.create(
            thread_id = self.thread.id,
            assistant_id = self.assistant.id,
            instructions = instructions
        )

    # Wait for the assistant to complete processing and handle outputs
    def wait_for_completion(self):
        while True:
            time.sleep(5)
            run_status = self.client.beta.threads.runs.retrieve(
                thread_id = self.thread.id,
                run_id = self.run.id
            )

            if run_status.status == 'completed':
                self.process_messages()
                break
            elif run_status.status == 'requires_action':
                print("Assistant required calling the API to get the data.")
                self.call_required_functions(run_status.required_action.submit_tool_outputs.model_dump())
            else:
                print("Waiting for the Assistant to process the question.")

    # Process and print messages from the assistant
    def process_messages(self):
        messages = self.client.beta.threads.messages.list(thread_id = self.thread.id)
        messages.data = messages.data[::-1]

        for msg in messages.data:
            role = msg.role
            content = msg.content[0].text.value
            print(f"{role.capitalize()}: {content}")

    # Call required functions based on assistant's needs
    def call_required_functions(self, required_actions):
        tool_outputs = []

        for action in required_actions["tool_calls"]:
            func_name = action['function']['name']
            arguments = json.loads(action['function']['arguments'])

            if func_name == "get_real_estate_agent_listed_properties_by_id":
                output = get_real_estate_agent_listed_properties_by_id(funda_id = arguments['funda_id'])
                tool_outputs.append({
                    "tool_call_id": action['id'],
                    "output": json.dumps(output)
                })

            if func_name == "get_top_real_estate_brokers_with_the_most_amount_of_homes":
                output = get_top_real_estate_brokers_with_the_most_amount_of_homes()
                tool_outputs.append({
                    "tool_call_id": action['id'],
                    "output": json.dumps(output)
                })
            
            if func_name == "get_top_real_estate_brokers_with_the_most_amount_of_homes_with_garden":
                output = get_top_real_estate_brokers_with_the_most_amount_of_homes_with_garden()
                tool_outputs.append({
                    "tool_call_id": action['id'],
                    "output": json.dumps(output)
                })

            if func_name == "get_top_real_estate_brokers_with_the_most_amount_of_homes_with_balcony_or_terrace":
                output = get_top_real_estate_brokers_with_the_most_amount_of_homes_with_balcony_or_terrace()
                tool_outputs.append({
                    "tool_call_id": action['id'],
                    "output": json.dumps(output)
                })

            if func_name == "get_top_real_estate_brokers_with_the_most_amount_of_homes_with_garage":
                output = get_top_real_estate_brokers_with_the_most_amount_of_homes_with_garage()
                tool_outputs.append({
                    "tool_call_id": action['id'],
                    "output": json.dumps(output)
                })

        print("Submitting outputs back to the Assistant.")
        self.client.beta.threads.runs.submit_tool_outputs(
            thread_id = self.thread.id,
            run_id = self.run.id,
            tool_outputs = tool_outputs
        )

# Main function to initiate the conversation with the assistant
def main():
    print("Starting a conversation with the Assistant.")
    assistant_manager = AssistantManager(os.getenv("api_key"))

    created_assistants = assistant_manager.list_assistants()
    created_assistant = [assistant for assistant in created_assistants.data if assistant.name == 'Funda Real Estate Brokers Assistant 2']
    if created_assistant == []:
        print("Creating a new Funda assistant.")
        assistant_manager.create_funda_assistant()
    else:
        print("Using existing assistant:", created_assistant[0].name)
        assistant_manager.select_assistant(created_assistant[0].id)

    users_questions = ["Give the phone number of the real estate broker with the most homes.", 
                       "Please give me the top 10 real estate brokers that own the most homes with gardens.",
                       "Give me property listings of the broker with the most amount of balconies.",
                       "Give me property listing adresses of the broker with the most amount of garages.",
                       "When did a broker with the most houses add his last home?"]
    
    for question in users_questions:
        assistant_manager.create_thread()
        assistant_manager.add_message_to_thread(role = "user", content = question)
        assistant_manager.run_assistant(instructions = "Answer the user's question in a friendly and precise manner.")
        assistant_manager.wait_for_completion()

# Entry point for the script
if __name__ == '__main__':
    main()