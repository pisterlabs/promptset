#
#Program Name: AiBot_Twitter
#Author: Jyasi' Davis
#Purpose: To send DMs to 400 Twitter accounts in one day

# 1. Setup Environment Variables
# 2. Download queried usernames from DataMiner
# 3. Get bio from usernames, and validate matches
# 4. Generate personalized Chat GPT response using DM format 
# 5. Send DMs to 40 accounts, marking them as messaged, and add date stamp
# 6. Do this for 10 twitter accounts
# 7. BONUS: At 8pm, send me a message asking if the program should run again the next day.
#           If I don't respond, run program again.

#---- 1. Setup Environment Variables -------------------------------------------------------------------------
# Required Libraries
import requests
import time
import tweepy
import openai
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Set up Google Sheets API credentials
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('ai-bot-twitter-08dd107ad8e6.json', scope)
client = gspread.authorize(creds)

#---- 2. Download queried usernames from DataMiner---------------------------------------------------------------
# Access the spreadsheet by its title or URL
spreadsheet = client.open('Twitter Accounts')

# Select a specific worksheet
worksheet = spreadsheet.worksheet('Sheet1')

# Get Twitter usernames from the spreadsheet
scrapedUsers = worksheet.get_all_records()

#---- 3. Get bio from usernames, and validate matches-------------------------------------------------------------
def get_bio_and_validate_matches(usernames, keyword):
	matched_usernames = []
	
	for username in usernames:
		try:
			user = api.get_user(screen_name=username)
			bio = user.description  # Get user's bio from Twitter
			if keyword in bio:
				matched_usernames.append(username)
		except tweepy.TweepError as e:
			print(f"Error fetching user data for {username}: {str(e)}")

	return matched_usernames

#----  4. Generate personalized Chat GPT response using DM format----------------------------------------
def generate_meeting_request_dm(account_username):
	# Define a structured message template
	template = {
		'intro': f"Hi {account_username}, I'm looking to connect with other indie game devs on Twitter and thought we could chat!",
		'social_proof': "I know this is random, but I actually specialize in boosting revenue using tailored funnels for game devs and streamers.",
		'mechanism': "One thing that makes us so different is we're so sure of our process we give you free ad spend.",
		'cta': "And more revenue means more dev time! Here's a quick run down on how we do it: [https://rb.gy/vaypj]",
	}
	
	# Replace placeholders in the template with the account's username
	for key, value in template.items():
		template[key] = value.format(account_username=account_username)
	
	# Combine the template steps into the full message
	full_message = "\n".join(template.values())
	full_message
	
	# Generate additional content using GPT-3
	response = openai.Completion.create(
		engine="davinci",
		prompt=full_message,
		max_tokens=100
	)
	generated_content = response.choices[0].text.strip()
	
	return generated_content

#----  5. Send DMs to 40 accounts, marking them as messaged, and add date stamp----------------------------------------
def send_dm_to_accounts(usernames, api):
	# Initialize total DM count
	total_dm_count_cell = worksheet.find("Total DM Count")
	total_dm_count = int(worksheet.cell(total_dm_count_cell.row, total_dm_count_cell.col + 1).value)
	currentDM_count = 0
	
	for username in usernames:
		if currentDM_count >= 40:  # Check if the daily send limit is reached
			break
	
		dm_text = generate_meeting_request_dm(username)  # Generate DM text
		try:
			api.send_direct_message(username, text=dm_text)  # Send DM
			print(f"Sent DM to {username}: {dm_text}")
	
			# Update total DM count
			currentDM_count += 1
	
			# Mark the account as messaged by adding 'X' next to the username in the spreadsheet
			cell_list = worksheet.findall(username)  # Find cells containing the username
			if cell_list:
				for cell in cell_list:
					# Get the row where the username is found and add 'X' in the adjacent cell
					row = worksheet.row_values(cell.row)
					row.append('X')
					row.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
					worksheet.insert_row(row, index=cell.row)
	
		except tweepy.TweepError as e:
			print(f"Error sending DM to {username}: {str(e)}")
	
	# Update the total DM count in the spreadsheet
	total_dm_count += currentDM_count
	worksheet.update(total_dm_count_cell.row, total_dm_count_cell.col + 1, total_dm_count)

#----  6. Do this for 10 Twitter accounts----------------------------------------
# Function to read Twitter account credentials from another sheet
def get_twitter_credentials(sheet, credentials_sheet_name):
	credentials_sheet = sheet.worksheet(credentials_sheet_name)
	data = credentials_sheet.get_all_records()
	credentials = []
	
	for row in data:
		credentials.append({
			"consumer_key": row["Consumer Key"],
			"consumer_secret": row["Consumer Secret"],
			"access_token": row["Access Token"],
			"access_token_secret": row["Access Token Secret"]
		})
	
	return credentials

# Name of the sheet where you store Twitter account credentials
credentials_sheet_name = 'Twitter Credentials'

# Get the list of Twitter credentials from the other sheet
twitter_accounts = get_twitter_credentials(spreadsheet, credentials_sheet_name)


# Main program
if __name__ == '__main__':
keyword = 'your_keyword'  # Replace with your target keyword
for _ in range(10):  # Repeat the process for 10 times
	# Get Twitter account credentials from your sheet (replace 'Twitter Credentials' with the correct sheet name)
	twitter_accounts = get_twitter_credentials(spreadsheet, 'Twitter Credentials')

	for account_credentials in twitter_accounts:
		# Set up code's client auth using Tweepy for the current account
		auth = tweepy.OAuthHandler(account_credentials["consumer_key"], account_credentials["consumer_secret"])
		auth.set_access_token(account_credentials["access_token"], account_credentials["access_token_secret"])
		api = tweepy.API(auth)

		matched_usernames = get_bio_and_validate_matches(scrapedUsers, keyword)  # Step 3
		send_dm_to_accounts(matched_usernames, api)  # Pass the API object for the current account

		time.sleep(60)  # Sleep for 60 seconds (adjust as needed)
