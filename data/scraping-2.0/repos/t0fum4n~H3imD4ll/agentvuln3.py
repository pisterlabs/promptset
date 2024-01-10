import requests
import json
import keys
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import argparse
from openai import OpenAI
openai_client = OpenAI(api_key=keys.openai_api_key)


# Suppress the InsecureRequestWarning from urllib3
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Wazuh API and Discord details, along with credentials
api_url = keys.wazapi_url
username = keys.wazusername
password = keys.wazpassword
discord_token = keys.discord_token
discord_channel_id = keys.discord_channel_id
chat_history_file = 'chat_history.json'


def get_jwt_token(api_url, username, password):
    auth_url = f"{api_url}/security/user/authenticate"
    response = requests.post(auth_url, auth=(username, password), verify=False)

    if response.status_code == 200:
        return response.json()['data']['token']
    else:
        print(f"Authentication failed with status code {response.status_code}")
        print("Response content:", response.text)
        return None


def get_vulnerabilities(api_url, agent_id, token):
    if not token:
        return None

    endpoint = f"/vulnerability/{agent_id}"
    url = f"{api_url}{endpoint}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, verify=False)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None


def summarize_vulnerabilities(vulns):
    # Construct a conversation history similar to the second script
    conversation = [
        {"role": "system", "content": "You are a cyber security focused chatbot. Your task is to summarize vulnerabilities into concise, readable summaries for a Discord channel."},
        {"role": "user", "content": f"Summarize these vulnerabilities: {json.dumps(vulns)}"}
    ]

    # Generate the summary with OpenAI's API
    response = openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=conversation
    )

    # Extract and return the summary
    summary = response.choices[0].message.content
    return summary

def format_vulnerabilities_for_discord(vulns, agent_id):
    # Generate the summary
    summary = summarize_vulnerabilities(vulns)

    # Format the summary for Discord
    formatted_message = f"Vulnerabilities for Agent {agent_id}:\n{summary}"
    return formatted_message


def send_discord_message_sync(channel_id, message):
    url = f"https://discord.com/api/v9/channels/{channel_id}/messages"
    headers = {
        "Authorization": f"Bot {discord_token}",
        "Content-Type": "application/json",
    }

    # Discord's character limit per message
    char_limit = 2000

    # Splitting the message into chunks
    for i in range(0, len(message), char_limit):
        chunk = message[i:i+char_limit]
        payload = {"content": chunk}
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code not in [200, 201]:
            print("Failed to send message to Discord. Status code:", response.status_code)

    print("Message sent to Discord successfully.")



def read_write_chat_history(message, filename=chat_history_file):
    try:
        with open(filename, 'r') as file:
            chathistory = json.load(file)
    except FileNotFoundError:
        chathistory = []

    chathistory.append({"role": "system", "content": message})

    with open(filename, 'w') as file:
        json.dump(chathistory, file, indent=4)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Get vulnerabilities for a specified Wazuh agent.")

    # Add an argument for agent_id
    parser.add_argument('agent_id', type=str, help='Agent ID to get vulnerabilities for')

    # Parse the arguments
    args = parser.parse_args()

    # Use the provided agent_id
    agent_id = args.agent_id

    token = get_jwt_token(api_url, username, password)
    vulns = get_vulnerabilities(api_url, agent_id, token)

    if vulns:
        message = format_vulnerabilities_for_discord(vulns, agent_id)
        send_discord_message_sync(discord_channel_id, message)
        read_write_chat_history(message)
    else:
        print("No data or error occurred.")

if __name__ == "__main__":
    main()

