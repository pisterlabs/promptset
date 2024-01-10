# Import necessary libraries
from flask import Flask, jsonify, request, session, render_template, url_for, redirect
from flask import Flask, session
from flask_session import Session
import json
import requests
from google.auth import exceptions
import gspread
from google.oauth2 import service_account
from google.auth.exceptions import GoogleAuthError
from datetime import datetime, timedelta
import random
import signal
import sys
from dotenv import load_dotenv
import openai  # Make sure to have OpenAI Python library installed
import os

#-------------------------------------------------------------------------------------------------------------------
# Step 1: Define Tool Functions and Environment Variables
#------------------------------------------------------------------------------------
def initialize_sheet(sheet_number):
    # Load credentials from the JSON file
    try:
        gc = gspread.service_account(filename='/home/ubuntu/aibot_twitter/ai-bot-twitter-08dd107ad8e6.json')
    except exceptions.GoogleAuthError as e:
        print()
        print(f"Error authenticating: {e}")
        return None

    sheet_name = "Prospected Usernames and Bot Accounts"

    try:
        sh = gc.open(sheet_name)
        worksheet = sh.get_worksheet(sheet_number)  # Use sheet1 or specify the desired sheet
        values = worksheet.get_all_values()
        return values

    except (gspread.SpreadsheetNotFound, gspread.WorksheetNotFound, IndexError):
        # If the sheet doesn't exist or if sheet_number is out of range
        print()
        print(f"Sheet with number {sheet_number} not found or out of range.")
        return None  # Return None or handle the case accordingly
        
def get_instagram_data(endpoint, params):
    try:
        base_url = 'https://graph.instagram.com/v12.0/'
        response = requests.get(f'{base_url}{endpoint}', params=params)
        
        # Print the response text for debugging
        print()
        print("Getting IG Data..")
        print()
        
        if response.status_code == 200 and response.text:
            return response.json()
        else:
            print()
            print("Failed to get data from Instagram. Status code:", response.status_code)
            print("IG Data Text:", response.text)
           
            # Print specific headers for more information
            print("WWW-Authenticate Header:", response.headers.get('WWW-Authenticate'))
            print("X-FB-Debug Header:", response.headers.get('X-FB-Debug'))
            print("IG-Api-Error-Message Header:", response.headers.get('IG-Api-Error-Message'))
            
            # Check if the response can be parsed as JSON and contains error_subcode
            try:
                error_json = response.json()
                if 'error_subcode' in error_json:
                    error_subcode = error_json['error_subcode']
                    print(f"Error Subcode: {error_subcode}")
            except json.JSONDecodeError:
                print()
                print("Error in decoding json, suberror code cant be found or read.")
                return None
    except requests.RequestException as e:
        print()
        print(f"Error making Instagram API request: {e}")
        print(f"Response text: {response.text}")
        return None

# Load environment variables from .env
load_dotenv()
global zyteAPI, zyte_creds_path
user_id = "test user id"  # Replace with your actual user_id
# Google sheets variables 
creds_sheet = initialize_sheet(0)# Use the index of the sheet (0 for code setup, 1 for bot accounts, 2 for posts, 3 for prospects)
#print("DEBUG: creds_sheet:", creds_sheet)
bots_sheet_data = initialize_sheet(1) #setup_file.get_worksheet(1).get_all_records()
posts_sheet_data = initialize_sheet(2)#setup_file.get_worksheet(2).get_all_records()
prospects_sheet_data = initialize_sheet(3) #setup_file.get_worksheet(3).get_all_records()
hashtags_sheet_data = initialize_sheet(4)

bot_app_id = bots_sheet_data[1][3]
bot_secret = bots_sheet_data[1][4]

openai.api_key = creds_sheet[0][1]
DIALOGFLOW_KEY_FILE = creds_sheet[1][1]
zyteAPI = creds_sheet[2][1]
zyte_creds_path = creds_sheet[3][1]
os.environ['REQUESTS_CA_BUNDLE'] = '/etc/ssl/certs/ca-certificates.crt'

# Suppress only the InsecureRequestWarning from urllib3 needed for SSL verification
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

#Setup residential proxy with Zenrows API
#print(f"Key set: {zrosAPI}")
#res_proxy = f"http://{zrowsAPI}:premium_proxy=true&proxy_country=us@proxy.zenrows.com:8001"
#res_proxies = {"http": res_proxy, "https": res_proxy}

# Initialize DialogFlow client
scope = ['https://www.googleapis.com/auth/dialogflow']
try:
    credentials = service_account.Credentials.from_service_account_file(
        DIALOGFLOW_KEY_FILE, scopes=scope
    )
    client = gspread.authorize(credentials)
except GoogleAuthError as e:
    print(f"Error initializing Google Sheets client: {e}")

# Define your Instagram accounts and proxy configurations
bots = []
for row in bots_sheet_data:
    bot = {
        "username": row[0],
        "password": row[1],
        "access_token": "{}|{}".format(bot_app_id, bot_secret),
        }
    
    bots.append(bot)

#Debug Auth
print()
print(f"Starting Bot.. with credentials /n(Secret: {bot_secret}, ID: {bot_app_id})")

#Get and return iguser id to get hashtag
try:
    params = {'fields': 'id', 'access_token': bots[0].get("Access_Token")}
    response = get_instagram_data('ig_user_id', params)  # Replace 'ig_user_id' with the actual endpoint

    if response and 'id' in response:
        user_id = response['id']
        bots_sheet_data[1][5] = user_id

except Exception as e:
    print()
    print(f"Error getting Instagram User ID: {e}")

print(f"Running bot as user: {user_id}")
print()
print()
print("Tool functions operational. Initializing application functions..")
print()
    

# Flask app for DialogFlow fulfillment
app = Flask(__name__)

# Set up a session for storing script and global status
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Placeholder variable for uncontacted and contacted usernames
#uncontacted_usernames = [...]  # Replace with actual data
#contacted_usernames = [...]  # Replace with actual data

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

#-------------------------------------------------------------------------------------------------------------------
# Step 2: Define functions
#----------------------------------------------------------------------------------------------------
def update_google_sheet(sheet_name, data):
    # Implement logic to update data in Google Sheets
    # Example: Use gspread library to update a sheet
    gc = gspread.service_account(filename='path/to/credentials.json')
    sh = gc.open(sheet_name)
    worksheet = sh.get_worksheet(0)  # Assumes data is stored in the first worksheet

    # Append the data to the worksheet
    worksheet.append_table([list(data.values())])

# Function to gracefully shutdown Flask server for code updates
def signal_handler(sig, frame):
    print('Shutting down gracefully...')
    # Perform cleanup tasks if necessary
    sys.exit(0)
    
#Function to post comment
def post_comment(post_id, context, access_token):
    # Instagram Graph API request to post a comment
    api_url = f"https://graph.instagram.com/v12.0/{post_id}/comments"
    
    # Customize the message based on the previous context
    template = {
        'intro': f"Hello again! {previous_message} Let's continue our conversation.",
        'book_meeting': "How about scheduling a meeting to discuss this further? It's 15-30 minutes, you can ask me any question, and I can actually answer them! You can pick a time that works for you here: [https://calendly.com/genusglobal/studios].",
    }

    # Combine the template steps into the full message
    full_message = f"{context} {template[context]}"

    # Format content by defining GPT-3 role when generating comment
    response = openai.Completion.create(
        engine="davinci",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': full_message}],
        max_tokens=100
    )
    generated_message = response.choices[0].text.strip()
    
    params = {'access_token': access_token, 'message': generated_message}
    response = requests.post(api_url, params=params)#post comment

    # Check for successful comment posting
    if response.status_code == 200:
        print(f"Comment posted successfully on post {post_id}")
    else:
        print(f"Error posting comment on post {post_id}. Status code: {response.status_code}")

#Fetch Hashtag IDs to do search
def get_hashtag_id(hashtag):
    #Get and Return Hashtag ID
    try:
        params = {'user_id': user_id, 'q': hashtag}
        response = get_instagram_data('ig_hashtag_search', params)

        if response and 'data' in response:
            hashtag_data = response['data']
            if hashtag_data:
                # Assuming the first result contains the desired hashtag ID
                return hashtag_data[0].get('id')

    except Exception as e:
        print(f"Error converting hashtag to ID: {e}")
        return None


#KPI 1 - Get posts
def search_posts_by_hashtag(hashtag):
    try:
        # Get hashtag ID
        hashtag_id = get_hashtag_id(hashtag)
        update_global_status(f"Searching hashtag: {hashtag_id}")

        # Store hashtag ID in Google Sheets
        if hashtag_id:
            hashtags_sheets_data[1][3] = hashtag_id

        # Search using the hashtag ID
        params = {'q': hashtag_id, 'access_token': bots[0].get("Access_Token")}
        return get_instagram_data('ig_hashtag_search', params)

    except Exception as e:
        print(f"Error searching posts by hashtag: {e}")
        return None

#KPI 2 - Get prospects from posts
def process_comments(media_id, keyword, access_token):
    global prospect_username  # Assuming you have a global variable for the Google Sheet

    # Fetch comments for a given media id
    comments_data = get_instagram_data(f'{media_id}/comments', {'access_token': access_token})

    # Process comments and update Google Sheets when keyword is found in the user's bio
    for comment in comments_data['data']:
        username = comment['username']

        # Fetch user bio using Instagram Graph API
        user_data = get_instagram_data(username, {'fields': 'biography', 'access_token': access_token})
        user_bio = user_data.get('biography', 'Bio not available')

        # Check if the keyword is in the user's bio
        if keyword in user_bio.lower():
            prospect_username = username
            # Update prospects sheet using the global variable (replace this with your actual logic)
            update_google_sheet(prospects_sheet_data, username)
            update_global_status(f"Updating prospects sheet with Username: '{username}', Bio: '{user_bio}'.")
            
#KPI 3 - Tier 1 Outreach
def generate_comments_and_mark_contacted(username):
    # Implement logic to generate comments and mark as contacted
    global outreach_done
    # Example: Fetch user posts, generate comments, and mark as contacted
    user_posts = get_instagram_data(f'{username}/media', {'access_token': 'your_access_token'})
    
    # Pick three random posts
    selected_posts = random.sample(user_posts, 3)
    post1 = selected_posts[0]
    post2 = selected_posts[1]
    post3 = selected_posts[2]
    
    # Leave a comment on one post
    comment_text = get_instagram_data(f'{post1[id]}/caption', {'access_token': 'your_access_token'})
    post_comment(post1['id'], comment_text, access_token)

    # Leave a call-to-action (CTA) on another post
    cta_text = get_instagram_data(f'{post2[id]}/caption', {'access_token': 'your_access_token'})
    post_comment(post2['id'], cta_text, access_token)

    # Schedule a comment for the third post
    scheduled_comment = get_instagram_data(f'{post3[id]}/caption', {'access_token': 'your_access_token'})
    post_comment(post3['id'], scheduled_comment, access_token)

    #mark as contacted
    outreach_done += 1
    update_global_status(f"Tier 1 Outreach complete. Current Outreach: {outreach_done}")
    
    
#KPI 4 - Tier 2 Outreach
def check_and_respond_to_dm_inquiries(bot_account):
    messages = get_instagram_data(f'{bot_account}/messages', {'access_token': 'your_access_token'})
    for message in messages['data']:
        # Example: Use DialogFlow to check for inquiries and respond accordingly
        #Dialogflow implementation
        update_global_status("Feature undeveloped. Current logic sound.")

def follow_up_with_usernames(uncontacted_usernames, contacted_usernames):
    for username in uncontacted_usernames:
        user_posts = get_instagram_data(f'{username}/media', {'access_token': 'your_access_token'})
        # Example: Generate comments and schedule follow-ups

def post_ad_posts_with_tensorflow():
    # Implement logic to post ad posts with TensorFlow model
    # Example: Use TensorFlow model to generate ad posts and post them
    update_global_status("Feature undeveloped. Current logic sound.")

def generate_and_post_story():
    # Implement logic to generate and post a story
    # Example: Use Instagram Graph API to post a story
    update_global_status("Feature undeveloped. Current logic sound.")

def schedule_posts(posts_type, schedule_date):
    # Implement logic to schedule posts
    # Example: Schedule posts based on specified type and date
    update_global_status("Feature undeveloped. Current logic sound.")

#meetings using air.ai
def close_meetings():
    #run once a day 
    
    #Get google sheet with numbers
    
    #run air.ai for all numbers not proccessed
    
    #if lead purchased service, fulfill service
    #Service Fulfillment:
    
    #if lead scheduled again, verify booking 
    
    #else just mark notes
    
    #mark the messaged as contacted with result, and update control panel
    update_global_status("Meetings Closed.")
    

# Main script to execute the Instagram Graph API workflow
def instagram_graph_api_script():
    # Loop for 30 times (Step 7)
    # 7. KPI#1: Search recent posts by hashtag and store data in posts sheets
    for hashtag in hashtags_sheet_data:
        posts_data = search_posts_by_hashtag(hashtag)
            
        # Store relevant data in posts sheet
        if posts_data is not None and 'data' in posts_data:
            for post in posts_data.get('data', []):
                data_to_store = {
                    'media_id': post.get('id'),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'user_id': post.get('user', {}).get('id'),
                    'username': post.get('user', {}).get('username'),
                    'caption': post.get('caption', {}).get('text', '')
                }
                update_google_sheet('posts_sheet', data_to_store)
                # Set date to run again.
        else:
            print("Error: No data or invalid data from Instagram API")
            break

    update_global_status("Weekly API calls used. Initial prospecting complete.")
    
    # 8. KPI#2: Process comments and store prospects
    for post_data in posts_sheet_data:
        process_comments(post_data['media_id'], "keyword")
    update_global_status("Usernames generated. Ready for outreach")
    
    # 9. KPI#3: Generate comments and mark as contacted, outreach lvl 1
    for username in comment_sheet_data:
        generate_comments_and_mark_contacted(username)
    update_global_status("Outreach completed. Stats updated.")
    
    # 10. KPI#4: 4x a day, get messages from bot account and respond to inquiries
    for _ in range(4):
        check_and_respond_to_dm_inquiries(bots_sheet_data['bot_account'])
    update_global_status("Inbox processed.")

    # 11. For each uncontacted username, get two random posts, generate comment on one, and mark follow-up date
    #follow_up_with_usernames(uncontacted_usernames, contacted_usernames)

    # 12. Check if any contacted usernames have a follow-up
    # (Implementation depends on your specific logic for follow-ups)

    # 13. Post batch of ad posts with TensorFlow model
    #post_ad_posts_with_tensorflow()

    # 14. Generate and post story
    #generate_and_post_story()

    # 15. Check to see if new comments, posts, or stories need to be scheduled
    #schedule_posts("comments", datetime.now() + timedelta(days=1))
    #schedule_posts("posts", datetime.now() + timedelta(days=2))
    #schedule_posts("stories", datetime.now() + timedelta(days=random.randint(1, 3)))
            

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
        instagram_graph_api_script()

    # Update the session variable with the new script status
    session['script_enabled'] = script_enabled

    return redirect(url_for('control_panel'))

# Example: Function to handle the Dialogflow webhook request
@app.route('/dialogflow-webhook', methods=['POST'])
def dialogflow_webhook():
    req = request.get_json()

    # Placeholder code, replace with actual DialogFlow intent handling
    intent = req['queryResult']['intent']['displayName']
    if intent == 'Inquiry':
        # Implement logic for handling the specific intent, send booking link with chat GPT format
        fulfillment_text = 'Your fulfillment text here.'
    else:
        # Handle other intents if needed
        fulfillment_text = 'Default fulfillment text.'

    return jsonify({'fulfillmentText': fulfillment_text})


# Start the Flask server
if __name__ == '__main__':
    # Set up a signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    app.run(host='0.0.0.0', port=80)  # Start the Flask server for DialogFlow request fulfillment
    # Run the Instagram Graph API script
    instagram_graph_api_script()
