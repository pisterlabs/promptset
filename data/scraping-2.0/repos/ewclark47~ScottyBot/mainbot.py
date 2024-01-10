"""
This script initializes a Slack Bot and a Mircrosoft Azure LUIS project. The Bot
then performs one of various functions based on user input whenever a user in the
Slack channel types @ScottyBot with a message. This script pulls out the pertinent
message and user information to then pass on to the LUIS module and the database 
module. Once the database and LUIS modules have performed their functions then
the Bot will respond based on what function was completed.
The functions that ScottyBot currently performs are:
AddCourse - add a course to a users schedule based on courseNumber
DropCourse - removes a course from a users schedule based on courseNumber
FindCourse - searches available courses and tries to match what the user is requesting
ViewSchedule - displays the users current schedule
"""

# imports
from os import environ
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from LUIS_functions import LUIS
import database_functions
import openai
from gpt3 import gpt3
#from twilio.twiml.messaging_response import MessagingResponse

# set up the environment 
load_dotenv()
SLACK_BOT_TOKEN = environ.get('SLACK_BOT_TOKEN')
SLACK_APP_TOKEN = environ.get('SLACK_APP_TOKEN')
clu_endpoint = environ.get("AZURE_CONVERSATIONS_ENDPOINT")
clu_key = environ.get("AZURE_CONVERSATIONS_KEY")
project_name = environ.get("AZURE_CONVERSATIONS_PROJECT_NAME")
deployment_name = environ.get("AZURE_CONVERSATIONS_DEPLOYMENT_NAME")
openai.api_key = environ.get("OPENAI_API_KEY")

# intialize the bolt app for ScottyBot and the LUIS functionality
app = App(token = SLACK_BOT_TOKEN)
luis = LUIS(clu_endpoint, clu_key, project_name, deployment_name)
# initialize gpt3 functionality
gpt3 = gpt3(openai.api_key)

# LUIS response and database functionality based on LUIS determined intent
def switch(action, categories, values, username, message):
	luisResponse="Sorry, I didn't quite get that"
	if action == "AddCourse":
		# then go through entity categorys to see if there is a course number, if not request it
		for i in range(len(categories)):
			if categories[i]=='CourseNumber':
				database_functions.addCourse(username, values[i])
				luisResponse = luis.addCourse(values[i])
			else:
				luisResponse = "Please try again, I didn't quite get that course number"
	elif action == "DropCourse":
		# then go through entity categorys to see if there is a course number, if not request it
		for i in range(len(categories)):
			if categories[i]=='CourseNumber':
				database_functions.dropCourse(username, values[i])
				luisResponse = luis.dropCourse(values[i])
			else:
				luisResponse = "Please try again, I didn't quite get that course number"
	elif action == "FindCourse":
		# go through entities and find a topic or course number to pass on to LUIS
		for i in range(len(categories)):
			if categories[i]=='CourseNumber' or categories[i]=='Topic':
				possCourses = database_functions.findCourse(values[i])
				luisResponse = luis.findCourse(possCourses)
			else:
				luisResponse = "Please try again, I didn't quite get that"
		if not (len(categories)>0):
			possCourses = database_functions.findCourse(" ")
			luisResponse = luis.findCourse(possCourses)
	elif action == "ViewSchedule":
		#scheduleString = "This will be replaced by info from the database"
		scheduleString = database_functions.viewSchedule(username) # database functions not fully ready
		luisResponse = luis.viewSchedule(scheduleString)
	else: # eventually this will be an elif for the Help action and small talk will be the else
		#luisResponse = luis.aboutBot()
		print("Calling gpt3")
		luisResponse = gpt3.gpt_response(message)
	return luisResponse

@app.event("app_mention")
def mention_handler(body, say):
	event = body['event'] # pulls the part of the body dictionary that contains the user message
	text = event['text'] 
	user = "<@" + event['user'] + ">"
	username=user[2:-1] # the users name without the @ symbol for use in the database
	message = text[15:] # this is just the user input message without the bot name
	# the above is the full uterance
	#print(event) 
	#print(message)
	
	# store the intent that LUIS determined from user input
	luisPrediction = luis.getLuisResults(message)
	luis.displayResults(message)

	# store all LUIS discovered entity categories and their values i.e. category: CourseNumber, value: 95729
	action = luisPrediction["result"]["prediction"]["topIntent"]
	entity_cats = []
	entity_vals = []
	for entity in luisPrediction["result"]["prediction"]["entities"]:
		entity_cats.append(entity["category"])
		entity_vals.append(entity["text"])
        
	bot_response = switch(action, entity_cats, entity_vals, username, message)
	say(bot_response + "\n" + user)  # user variable holds the @name of whoever initiated the bot, use everything after the @ as the username for the db calls

# maybe add an on startup so that the user can get a run down of what ScottyBot does

if __name__ == "__main__":
	handler = SocketModeHandler(app, SLACK_APP_TOKEN)
	handler.start()

"""
The intro to the following tutorial was not working out:
https://www.analyticsvidhya.com/blog/2018/03/how-to-build-an-intelligent-chatbot-for-slack-using-dialogflow-api/
so the above code was found and modified from the following tutorial:
https://www.twilio.com/blog/how-to-build-a-slackbot-in-socket-mode-with-python#:~:text=To%20enable%20Socket%20Mode%2C%20navigate,your%20app%20is%20used%20in.
*note: the first tutorial had info on dialog flow that may be useful continuing on
"""