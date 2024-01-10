#-------------------------------------------------------------------------------------------------------------------
# Author: Jyasi' Davis
# Program Name: Aibot_Instagram-v4
#
# Program Purpose: To automate sending DM outreach, booking meetings with contacted ig accounts, closing 
#                   the meetings with Air.ai, and fulfilling the service
#-------------------------------------------------------------------------------------------------------------------
import requests
import sys
import json
import random
import gspread
import signal
import time
import openai
#import stylegan2
import os
import cv2
import numpy as np
from google.oauth2 import service_account
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from google.auth import credentials as google_auth_credentials
from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from flask import Flask, request, jsonify
from flask import render_template
from flask import Flask, render_template, redirect, url_for
from flask import Flask, render_template, redirect, url_for, session
from flask_session import Session
#from ig_bot import Bot
from dotenv import load_dotenv
from requests.packages.urllib3.exceptions import InsecureRequestWarning

#-------------------------------------------------------------------------------------------------------------------
# Step 1: Define Environment Variables
#-------------------------------------------------------------------------------------------------------------------
# Suppress only the InsecureRequestWarning from urllib3 needed for SSL verification
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Load environment variables from .env
load_dotenv()
global zyteAPI, zyte_creds_path
openai.api_key = os.environ.get("OPENAI_API_KEY")
creds_path = os.environ.get('GOOGLE_SHEETS_CREDS_PATH')  # Set this environment variable in your .env file
zrowsAPI = os.environ.get("ZENROWSAPIKEY")
zyteAPI = os.environ.get("zyteAPI")
zyte_creds_path = os.environ.get("ZYTEPATH")
DIALOGFLOW_KEY_FILE = os.environ.get("DIALOGFLOW_KEY_FILE")
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

# Set up Google Sheets API credentials
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

if creds_path:
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    client = gspread.authorize(creds)
else:
    print("Please set the GOOGLE_SHEETS_CREDS_PATH environment variable.")

# Open the Google Sheets sheet housing Instagram accounts and proxy configurations
spreadsheet_accounts = client.open('Prospected Usernames and Bot Accounts')
worksheet_accounts = spreadsheet_accounts.get_worksheet(1)  # Use the index of the sheet (0 for the first sheet, 1 for the second, and so on)

# Read Instagram accounts and proxy configurations from the worksheet
accounts_data = worksheet_accounts.get_all_records()

# Open the Google Sheets sheet housing prospects
spreadsheet_usernames = client.open('Prospected Usernames and Bot Accounts')  # Use the same document name
worksheet_usernames = spreadsheet_usernames.get_worksheet(0)  # Use the index of the sheet (0 for the first sheet, 1 for the second, and so on)

# Define your Instagram accounts and proxy configurations
accounts = []
for row in accounts_data:
    account = {
        "username": row["Username"],
        "password": row["Password"],
        "access_token": row["Access Token"],
        }
    
    accounts.append(account)

# Your Instagram Graph API access token
# Add more access tokens for your accounts if needed
access_tokens = [account["access_token"] for account in accounts]

#Setup residential proxy with Zenrows API
#print(f"Key set: {zrosAPI}")
res_proxy = f"http://{zrowsAPI}:premium_proxy=true&proxy_country=us@proxy.zenrows.com:8001"
res_proxies = {"http": res_proxy, "https": res_proxy}

# Initialize DialogFlow client
try:
    credentials = service_account.Credentials.from_service_account_file(
        DIALOGFLOW_KEY_FILE, scopes=scope
    )
    client = gspread.authorize(credentials)
except GoogleAuthError as e:
    print(f"Error initializing Google Sheets client: {e}")

#try:
    #credentials, _ = google_auth_credentials.default(DIALOGFLOW_KEY_FILE)
    #dialogflow_session_client = dialogflow.SessionsClient(credentials=credentials)
#except google_auth_credentials.GoogleAuthError as e:
    #print(f"Error initializing DialogFlow client: {e}")

# Set up Flask app for DialogFlow fulfillment
app = Flask(__name__)

# Configure the Flask app to use the Session middleware
app.config['SESSION_TYPE'] = 'filesystem'  # You can choose another session type if needed
Session(app)

# Define statistics variables
total_bookings = 0
outreach_done = 0
script_enabled = False
prospecting_limit = 5
global g_status
g_status = "Idle"

# Initialize a set to temporarily store prospected usernames
prospected_usernames = set()
prospecting_failed = False

# Variable to store the conversation context
conversation_context = []

# Define hashtags to search for
hashtags = ['indiegamedev', 'indiedev', 'gamedev', 'solodev']

#-------------------------------------------------------------------------------------------------------------------
# Step 2: Define functions
#-------------------------------------------------------------------------------------------------------------------
# Define a function to find and store a defined number unique usernames to Google Sheets document (Currently: 4000)
def find_and_store_usernames(account):
    #update_global_status("Debug message: Prospecting started...")
    global prospecting_failed
    prospecting_failed = False
    
    for _ in range(prospecting_limit):
        hashtag = random.choice(hashtags)
        next_url = f'{https://www.instagram.com/explore/search/keyword/?q={hashtag}'

        while next_url and len(prospected_usernames) < prospecting_limit and not prospecting_failed:
            try:
                # Implement code to find usernames and store them in Google Sheets and the set
                #session = requests.Session()
                #session.proxies = res_proxies
                #response = session.get(next_url, proxies=res_proxies, verify=False)

                response = requests.get(
                    next_url,
                    proxies={
                        f"{scheme}": f"http://{zyteAPI}:@api.zyte.com:8011/"
                        for scheme in ("http", "https")
                    },
                    verify=zyte_creds_path,
                )
                
                if response.status_code == 200:
                    #http_response_body: bytes = response.text
                    #print(http_response_body.decode())
                    #data = json.loads(http_response_body)
                    data = response.json()
                    if 'data' in data:
                        for post in data['data']:
                            if 'username' in post.get('caption', {}):
                                #Debug
                                update_global_status("Found ig user: " + username)
                                
                                username = post['caption']['username']
                                bio = get_user_bio(username)
                                keywords = ["indie game dev", "game dev"]
                                bio_lower = bio.lower()
                                if any(keyword in bio_lower for keyword in keywords):
                                    # Add the username to the set
                                    prospected_usernames.add(username)
                                    # Add the username to Google Sheets
                                    worksheet_usernames.append_row([username])  # You can append additional information as needed
                    next_url = data['paging'].get('next')
                else:
                    print()
                    print(f"Failed to fetch post data. Status Code: {response.status_code}")
                    print(f"Message: {response.text}")
                    print()
                    prospecting_failed = True
                    #update_global_status(f"Debug message: Prospecting process has ended. {len(prospected_usernames)} prospects found.")
                    #print()
                    break
            except Exception as e:
                prospecting_failed = True
                #update_global_status(f"Debug message: Prospecting process has ended. {len(prospected_usernames)} prospects found.")
                print(f"An error occurred: {str(e)}")
                print()
                print(f"Response: {response.text}")
                print()
                break
    
    if prospecting_failed == True:
        update_global_status(f"Debug message: Prospecting process has failed. {len(prospected_usernames)} prospects found.")
        print()
        global script_enabled 
        script_enabled = not script_enabled
    else:
        update_global_status(f"Debug message: Prospecting process has ended. {len(prospected_usernames)} prospects found.")
        print()

# Function to send a customized DM using ig username, bot, and residential proxy to a prospected username to book a meeting
def send_dm(username, account):
    session = requests.Session()

    # Set up proxy for this request
    session.proxies = res_proxies

    # Define a structured message template
    template = {
        'intro': f"Hi {username}, I'm looking to connect with other indie game devs on Instagram and thought we could chat!",
        'social_proof': "I know this is random, but I actually specialize in boosting revenue using tailored funnels for game devs and streamers.",
        'mechanism': "One thing that makes us so different is we're so sure of our process we give you free ad spend.",
        'cta': "And more revenue means more dev time! Here's a quick run down on how we do it: [https://rb.gy/vaypj]",
    }

    # Replace placeholders in the template with the account's username
    for key, value in template.items():
        template[key] = value.format(username=username)

    # Combine the template steps into the full message
    full_message = "\n".join(template.values)

    # Generate additional content using GPT-3
    response = openai.Completion.create(
        engine="davinci",
        prompt=full_message,
        max_tokens=100
    )
    generated_message = response.choices[0].text.strip()

    # Construct the DM data
    dm_data = {
        'recipient_user_id': username,
        'message': generated_message
    }

    # Send the DM using the Instagram Graph API
    response = session.post(f'https://graph.instagram.com/v13.0/me/media/abc123/messages?access_token={account["access_token"]}', json=dm_data)

    if response.status_code == 200:
        print(f'Sent DM to {username}: {generated_message}')
        cell = worksheet_usernames.find(username)
        worksheet_usernames.update_cell(cell.row, cell.col + 1, 'Messaged')
        return True
    else:
        print(f'Failed to send DM to {username}: {response.text}')
        return False


# Sends the DMS to all unproccessed prospects in google sheets file
def process_usernames():
    #update_global_status("Started Outreach..")
    global outreach_done
    usernames = worksheet_usernames.col_values(1)  # Assuming usernames are in the first column
    contacted = worksheet_usernames.col_values(2)
    
    for username, contacted_status in zip(usernames, contacted):
        if username != '' and contacted_status != 'Messaged':
            dm_count = 0  # Reset the dm_count for each new username
            for account in accounts:
                if dm_count >= 400:
                    send_dm(username, account)
                    dm_count += 1
                    outreach_done += 1
                    time.sleep(60)  # Sleep to respect Instagram's rate limits
                    break
    
    update_global_status(f"Outreach ended. Total outreach = {outreach_done}")
                    
# Function to create and post a reel (TO BE COMPLETED!)
#def create_and_post_reel(bot, username, password, proxy_info):
    # Log in to an Instagram account
    #bot.login(username=username, password=password, proxy=proxy_info)

    # Generate an image using StyleGAN2
    #latent_vector = stylegan2.run.generate_latent()
    #image = Gs.run(latent_vector)

    # Generate a caption using ChatGPT
    #generated_text = generate_text_with_gpt("Your prompt here")

    # Add the generated text as an overlay to the image
    #image_with_overlay = add_text_overlay(image, generated_text)

    # Convert the image to a frame and add it to the video
    #frame = cv2.cvtColor(np.array(image_with_overlay), cv2.COLOR_RGB2BGR)
    #video_writer.write(frame)

    # Release the VideoWriter
    #video_writer.release()

    #video_path = 'output_video.mp4'  # Provide the path to your generated video

    # Upload the video to Instagram
    #bot.upload_reel(video_path, caption=generated_text)

    # Log out from the account
    #bot.logout()
    
# Function to generate a personalized message using ChatGPT, maintains conversation during DialogFlow conversation. Always tries to book a meeting.
def generate_personalized_message(previous_message):
    # Store the previous message in the conversation context
    global conversation_context
    conversation_context.append(previous_message)

    # Customize the message based on the previous context
    template = {
        'intro': f"Hello again! {previous_message} Let's continue our conversation.",
        'book_meeting': "How about scheduling a meeting to discuss this further? It's 15-30 minutes, you can ask me any question, and I can actually answer them! You can pick a time that works for you here: [https://calendly.com/genusglobal/studios].",
    }

    # Combine the template steps into the full message
    full_message = "\n".join(template.values())

    # Generate additional content using GPT-3
    response = openai.Completion.create(
        engine="davinci",
        messages=conversation_context,  # Include the entire conversation context
        max_tokens=100
    )
    generated_message = response.choices[0].text.strip()

    return generated_message

# Function to extract information from the PDF file (using pdfplumber as an example), for fulfillment
#def extract_pdf_info(pdf_file_url):
    #import pdfplumber
    #import requests

    # Download the PDF file
    #response = requests.get(pdf_file_url)
    #with open('temp.pdf', 'wb') as temp_pdf:
        #temp_pdf.write(response.content)

    # Extract information using pdfplumber
    #with pdfplumber.open('temp.pdf') as pdf:
        # Add your logic here to extract relevant information
        # For example, extracting text from the PDF
        #text = ''
        #for page in pdf.pages:
            #text += page.extract_text()

    #return {'text': text}

#Close meetings using air.ai
def close_meetings():
    #run once a day 
    
    #Get google sheet with numbers
    
    #run air.ai for all numbers not proccessed
    
    #mark the messaged as contacted with result, and update control panel
    update_global_status("Meetings Closed.")
    
# Define your job to run your script, posts 1 reel for each bot account stored, prospects leads, and contacts them to book
def run_script():
     if script_enabled:
        #bot = Bot()
        #for account in accounts:
            #create_and_post_reel(bot, account, account["proxy"])
        #update_global_status("Starting prospecting..")
        find_and_store_usernames(accounts[0])
        if not prospecting_failed:
            update_global_status("Prospecting complete, starting outreach")
            process_usernames()
            #update_global_status("Outreach complete. Run again?")
            close_meetings()

# Function to handle the Dialogflow webhook request
@app.route('/dialogflow-webhook', methods=['POST'])
def dialogflow_webhook():
    req = request.get_json()
    
    # Extract intent and parameters from the Dialogflow request
    intent = req['queryResult']['intent']['displayName']

    if intent == 'BookMeeting':
        # Use ChatGPT to generate a personalized message asking for meeting details
        personalized_message = generate_personalized_message(req['queryResult']['queryText'])

        return jsonify({
            'fulfillmentText': personalized_message
        })
    elif intent == 'CollectMeetingDetails':
        # Extract parameters provided by the user
        date = req['queryResult']['parameters']['date']
        time = req['queryResult']['parameters']['time']
        location = req['queryResult']['parameters']['location']
        bookings += 1
       
        # You can now use the collected parameters to book the meeting and provide a response. (TODO!! ADD CALENDLY INTEGRATION TO BOOK MEETING)
        return jsonify({
            'fulfillmentText': f'Great! We have scheduled a meeting on {date} at {time} at {location}.'
        })
    else:
        # Handle other intents here if needed
        return jsonify({'fulfillmentText': "I am not sure how to respond to that, but it's always easier to talk in person. Just click here, and pick the best time and method that works for you! Hope to chat soon: https://calendly.com/genusglobal/studios."})

#Function to gracefully shutdown Flask server for code updates
def signal_handler(sig, frame):
    print('Shutting down gracefully...')
    # Perform cleanup tasks if necessary
    sys.exit(0)
#-------------------------------------------------------------------------------------------------------------------
# Step 3: Define routes for your control panel
#-------------------------------------------------------------------------------------------------------------------
# Define route to display the control panel
@app.route('/control_panel')
def control_panel():
    # Get actual data, e.g., script status, meetings booked, outreach count
    #script_status = "Off"  # Replace with actual script status
    
    # Get the script status from the session variable
    script_status = session.get('script_enabled', False)
    global_status = session.get('global_status', g_status)

    meetings_booked = 0  # Replace with actual data
    outreach_count = outreach_done  # Replace with actual data
    return render_template('control_panel.html', script_status=script_status, meetings_booked=meetings_booked, outreach_count=outreach_count, global_status=global_status)

# Example route to update the global status
@app.route('/update_global_status/<status>')
def update_global_status(status):
    # Update the global status
    g_status = status
    print(g_status)
    session['global_status'] = g_status
    return redirect(url_for('control_panel'))

#route to send shutdown signal from control panel 
@app.route('/shutdown', methods=['POST'])
def shutdown():
    print("Shutting down gracefully...")
    os.kill(os.getpid(), signal.SIGINT)
    return 'Server shutting down...'

@app.route('/toggle_script', methods=['POST'])
def toggle_script():
    global script_enabled  # Declare script_enabled as global

    # Retrieve the current script status from the session variable
    script_enabled = session.get('script_enabled', False)

    # Toggle the script status
    script_enabled = not script_enabled
    
    # Run the script if it's enabled
    if script_enabled:
        run_script()

    # Update the session variable with the new script status
    session['script_enabled'] = script_enabled

    return redirect(url_for('control_panel'))

@app.route('/increase_outreach', methods=['POST'])
def increase_outreach():
    global prospecting_limit
    if script_enabled:
        prospecting_limit += 1
    return redirect(url_for('control_panel'))

# Function to handle service fulfillment webhook. This is for after meetings close. Extracts based off of email to fulfull service. TODO!
@app.route('/service-fulfillment', methods=['POST'])
def service_fulfillment():
    req = request.get_json()

    # Extract information from the received form
    client_name = req['clientName']
    pdf_file_url = req['pdfFileUrl']

    # Add your logic here to customize DMs based on client information and PDF file
    # You can integrate this logic with your existing functions or create new ones

    # Example: Extract information from the PDF file
    pdf_info = extract_pdf_info(pdf_file_url)
    print(f"PDF Information: {pdf_info}")

    # Example: Customize DMs based on the extracted information and send them
    customize_and_send_dm(client_name, pdf_info)

    # Return a response indicating successful fulfillment
    return jsonify({'message': 'Service fulfilled successfully'})

#-------------------------------------------------------------------------------------------------------------------
# Step 4: Run the program
#-------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Set up a signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    app.run(host='0.0.0.0', port=80)  # Start the Flask server for DialogFlow request fulfillment
