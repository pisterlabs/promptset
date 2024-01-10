import requests
import random
import gspread
import time
import openai
from flask import Flask, request, jsonify
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Define your Instagram accounts and proxy configurations
# ... (the same as in your original code)

# Define your Dialogflow webhook server using Flask
app = Flask(__name__)

# Define your Instagram Graph API access token and credentials
# ... (the same as in your original code)

# Function to validate and store usernames based on bio
# ... (the same as in your original code)

# Function to send a DM using a specific account and proxy
# ... (the same as in your original code)

# Function to mark a username as "messaged" in the Google Sheets document
# ... (the same as in your original code)

# Function to handle the Dialogflow webhook request
@app.route('/dialogflow-webhook', methods=['POST'])
def dialogflow_webhook():
    req = request.get_json()
    
    # Extract intent and parameters from the Dialogflow request
    intent = req['queryResult']['intent']['displayName']
    params = req['queryResult']['parameters']

    if intent == 'SendDM':
        username = params['username']  # Extract the username from Dialogflow parameters
        # Find the corresponding account and proxy
        for account in accounts:
            if account['username'] == params['account']:
                account_info = account
                break
        else:
            return jsonify({'fulfillmentText': 'Sorry, I cannot find the specified account.'})

        # Send DM using the specified account and proxy
        success = send_dm(username, account_info['access_token'], account_info['proxy'])

        if success:
            # Mark the username as "messaged" in the Google Sheets document
            mark_as_messaged(username)
            return jsonify({'fulfillmentText': f'Sent DM to {username}.'})
        else:
            return jsonify({'fulfillmentText': f'Failed to send DM to {username}.'})

    # Handle other intents here if needed

# Main program
if __name__ == '__main__':
    # Start the Flask server on your AWS EC2 instance
    app.run(host='0.0.0.0', port=80)