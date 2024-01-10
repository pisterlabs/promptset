import jerkbot as jb
import openai

openai.api_key = "API_KEY"
server = "https://matrix.org" #not tested with other servers
token = "TOKEN" #Matrix access token found somewhere on settings page in element
username = "USERNAME"
password = "PASSWORD"
channel = "#CHANNEL:SERVER.org"  
personality = "a sarcastic jerk"

#create AIbot and start it
bot = jb.AIbot(server, token, username, password, channel, personality)
bot.start()

input("BOT RUNNING.  Press Enter to exit") #prevents the program from exiting
    
#bot.stop()
''' If uncommented, the bot leaves the channel when the program exits.
    disabled due to a bug where if you create a channel with the account you use for this bot,
    and nobody else joins the channel, and then you press enter to exit the program,
    the channel will be empty and you can't join it again.
    
    Commented out, the bot's username will remain in the channel even if the bot is not running.
'''
   
