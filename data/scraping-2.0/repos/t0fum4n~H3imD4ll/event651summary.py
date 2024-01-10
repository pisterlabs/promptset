import requests
import discord
import keys
import json
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from requests.auth import HTTPBasicAuth
from openai import OpenAI

# Suppress the InsecureRequestWarning from urllib3
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Elasticsearch and Discord details, along with credentials
es_host = keys.elasticsearch_host
index = 'wazuh-alerts-*'
rule_id = 651
discord_token = keys.discord_token
discord_channel_id = keys.discord_channel_id
es_username = keys.es_user
es_password = keys.es_pass
openai_client = OpenAI(api_key=keys.openai_api_key)

# Set up Discord intents and client
intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
discord_client = discord.Client(intents=intents)

# Initialize chat history
chat_history_file = 'alerts.json'

# Function definitions
def read_chat_history_from_file(filename=chat_history_file):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []  # Return an empty list if the file doesn't exist

def write_chat_history_to_file(chathistory, filename=chat_history_file):
    with open(filename, 'w') as file:
        json.dump(chathistory, file, indent=4)

def get_latest_alert(es_host, index, rule_id):
    # Construct the search query for events in the last 60 seconds
    query = {
        "query": {
            "bool": {
                "must": [{"match": {"rule.id": rule_id}}],
                "filter": [{"range": {"timestamp": {"gte": "now-60s"}}}]  # Modified time range
            }
        },
        "sort": [{"timestamp": {"order": "desc"}}],
        "size": 1  # If you want all events, not just the latest, you may increase this size or remove this line
    }

    # Specify the correct Content-Type header for Elasticsearch 8.x
    headers = {
        'Content-Type': 'application/json'
    }

    # Construct the full URL to the Elasticsearch _search endpoint
    search_url = f"{es_host}/{index}/_search"

    try:
        # Make the HTTP request to Elasticsearch
        response = requests.get(
            search_url,
            auth=(es_username, es_password),
            headers=headers,
            data=json.dumps(query),
            verify=False  # Only use this for development or testing
            # For production, use `verify='/path/to/cacert.pem'` to specify the CA bundle.
        )
        response.raise_for_status()  # Raise an exception for HTTP error codes

        # Parse the response JSON and extract the hits
        hits = response.json()['hits']['hits']
        return hits  # Return all hits
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Oops: Something Else, {err}")

def summarize_event(event):
    # Use OpenAI's chat completion API to summarize the event
    conversation = [
        {"role": "user", "content": f"""Summarize this alert with this format.

ALERT!! (type of attack detected) on agent (agent.id) from (country) at (timestamp)

Description: (Give a detailed but very brief account of the alert. Be sure to be descriptive but to the point)

Recommendation: (Give some short recommendations that the user can ask for more detail about later on.)

Additional details: (Put any other information here that would be relevant to this alert.)

##Alert summary##

Agent ID: 
User account:
Country:
Timestamp: 

    {json.dumps(event, indent=2)}"""}
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=conversation
    )

    summary = response.choices[0].message.content
    return summary

def send_discord_message_sync(channel_id, message):
    url = f"https://discord.com/api/v9/channels/{channel_id}/messages"
    headers = {
        "Authorization": f"Bot {discord_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "content": message
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        print("Message sent to Discord successfully.")
    else:
        print("Failed to send message to Discord.")

def main():
    chathistory = read_chat_history_from_file()  # Load the existing chat history
    events = get_latest_alert(es_host, index, rule_id)
    if events:
        for event in events:
            summary = summarize_event(event['_source'])  # Ensure this matches the structure of your event data
            send_discord_message_sync(discord_channel_id, summary)
            chathistory.append({"role": "system", "content": summary})
        write_chat_history_to_file(chathistory)
        print(f"{len(events)} events processed and written to chat history.")
    else:
        print("No recent brute force blocks")




# Run the main function
main()