import json
import httpx
import os
from openai import OpenAI
from prompts import assistant_instructions
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# OPENAI_API_KEY = process.env.OPENAI_API_KEY
# process.env.
AIRTABLE_API_KEY = os.getenv('AIRTABLE_API_KEY')
TELEGRAM_BOT_KEY = os.getenv('TELEGRAM_BOT_KEY')
TELEGRAM_CHANNEL_ID = os.getenv('TELEGRAM_CHANNEL_ID')


# Init OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

def find_message_with_no_assistant(messages):
    # Iterate through each message in the data list
    for message in messages.data:
        # Check if the assistant_id is None
        if message.assistant_id is None:
            # Return the message or its content as needed
            return message

    # Return None if no message with assistant_id=None is found
    return None

def ping_telegram(status, messages):
    # print ("messages: ", messages)

    # if messages is string

    if isinstance(messages, str):
      user_input = messages
    else:
      user_input = find_message_with_no_assistant(messages)
      if user_input:
          user_input = user_input.content[0].text.value
      else:
          user_input = "No user input found"

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_KEY}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHANNEL_ID,
        "text": f"Location: Instagram\n\nStatus: {status}\n\nUser Input: {user_input}",
    }
    response = httpx.post(url, data=payload)
    print("Telegram response:", response)
    return response.json()


# Add lead to Airtable
def create_lead(name, phone):
  url = "https://api.airtable.com/v0/appZtfsCkZGK0MWG4/Tables"
  headers = {
      "Authorization":
      AIRTABLE_API_KEY,  # NOTE: When adding your Airtable API key in secrets it must include "Bearer YOURKEY", keeping the Bearer and the space. If you don't add this then it won't work!
      "Content-Type": "application/json"
  }
  data = {"records": [{"fields": {"Name": name, "Phone": phone}}]}
  response = httpx.post(url, headers=headers, json=data)
  if response.status_code == 200:
    print("Lead created successfully.")
    return response.json()
  else:
    print(f"Failed to create lead: {response.text}")


# Create or load assistant
def create_assistant(client):
  assistant_file_path = 'assistant.json'

  # If there is an assistant.json file already, then load that assistant
  if os.path.exists(assistant_file_path):
    with open(assistant_file_path, 'r') as file:
      assistant_data = json.load(file)
      assistant_id = assistant_data['assistant_id']
      print("Loaded existing assistant ID.")
  else:
    # If no assistant.json is present, create a new assistant using the below specifications

    # To change the knowledge document, modify the file name below to match your document
    # If you want to add multiple files, paste this function into ChatGPT and ask for it to add support for multiple files
    # how to allow for a document hosted on the internet?
    file = client.files.create(file=open("dutch-knowledge.docx", "rb"),
                               purpose='assistants')

    assistant = client.beta.assistants.create(
        # Change prompting in prompts.py file
        instructions=assistant_instructions,
        model="gpt-4-1106-preview",
        tools=[
            {
                "type": "retrieval"  # This adds the knowledge base as a tool
            },
            {
                "type": "function",  # This adds the lead capture as a tool
                "function": {
                    "name": "create_lead",
                    "description":
                    "Capture lead details and save to Airtable.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Full name of the lead."
                            },
                            "phone": {
                                "type":
                                "string",
                                "description":
                                "Phone number of the lead including country code."
                            }
                        },
                        "required": ["name", "phone"]
                    }
                }
            }
        ],
        file_ids=[file.id])

    # Create a new assistant.json file to load on future runs
    with open(assistant_file_path, 'w') as file:
      json.dump({'assistant_id': assistant.id}, file)
      print("Created a new assistant and saved the ID.")

    assistant_id = assistant.id

  return assistant_id
