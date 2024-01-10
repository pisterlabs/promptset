import base64
import subprocess
import os
import openai
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from flask import Flask, request, jsonify
import re

openai_key = os.environ['OPENAI_KEY']
confluence_api_token = os.environ['CONFLUENCE_API_TOKEN']
slack_api_token = os.environ['SLACK_API_TOKEN']

def install_dependencies():
    packages = ['llama_index', 'openai', 'requests', 'beautifulsoup4', 'joblib', 'slack_sdk', 'pyngrok']
    installed_packages = subprocess.check_output(['pip3', 'freeze']).decode('utf-8').splitlines()

    missing_packages = [pkg for pkg in packages if pkg not in installed_packages]
    if missing_packages:
        with open(os.devnull, 'w') as null:
            subprocess.check_call(['pip3', 'install'] + missing_packages, stdout=null, stderr=subprocess.STDOUT)
        print("Installed missing dependencies:", missing_packages)
    else:
        print("All dependencies are already installed.")

# Install dependencies
install_dependencies()

print("Dependencies completed! Data Extraction occurring!")

# Import necessary packages
from llama_index import Document


## OpenAI Configuration ##
openai.api_key = openai_key

# Set up Confluence API details - using Test Page
base_url = "https://greg-carroll.atlassian.net/"
api_token = confluence_api_token
page_id = 98442

# Encode API token in Base64
credentials = f'{api_token}'
base64_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')

# Fetch page content from Confluence REST API
url = f'{base_url}/wiki/rest/api/content/{page_id}?expand=body.storage'
headers = {
    'Authorization': f'Basic {base64_credentials}',
    'Content-Type': 'application/json'
}
response = requests.get(url, headers=headers)

# Check if the response is successful
if response.status_code == 200:
    # Access the content based on the JSON structure
    try:
        content = response.json().get('body', {}).get('storage', {}).get('value', '')

        # Create a document from the content
        document = Document(text=content)

    except KeyError as e:
        print(f"KeyError: {e}")
else:
    print(f"Request failed with status code: {response.status_code}")

# Set up your Flask app
app = Flask(__name__)

# Set up the Slack API client
slack_token = slack_api_token
slack_client = WebClient(token=slack_token)

# Handle Slack events
@app.route('/slack/events', methods=['POST'])
def handle_slack_events():
    # Verify the request is coming from Slack
    if request.headers.get('X-Slack-Signature') and request.headers.get('X-Slack-Request-Timestamp'):
        # Process the event payload
        event_data = request.get_json()
        event_type = event_data['type']
        if event_type == 'event_callback':
            event = event_data['event']
            if event['type'] == 'message':
                # Handle the message event
                user_id = event['user']
                text = event['text']
                # Process the message and generate a response
                response = process_message(user_id, text)
                # Send the response back to the user
                try:
                    slack_client.chat_postMessage(channel=event['channel'], text=response)
                except SlackApiError as e:
                    print(f"Error sending message: {e.response['error']}")
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Invalid request'}), 400


# Bot Creation
def bot(index_files_path, document_text):
    vector_store = None  # Placeholder for the vector store
    vector_length = None  # Placeholder for the vector length

    while True:
        user_input = input('How can I help? ')
        # Query the document text using OpenAI
        response = openai.Completion.create(
            engine='text-ada-001',
            prompt=document_text + "\nUser: " + user_input + "\nBot:",
            max_tokens=50,
            temperature=0.6,
            n=1,
            stop=None,
            logprobs=0,
        )
        # Get the generated response from OpenAI
        generated_response = response.choices[0].text.strip()
        print(f"Response: {generated_response}\n")


if __name__ == '__main__':

    # Start the Flask app
    app.run(debug=True)