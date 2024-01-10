import os
import openai
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack import WebClient
from slack_bolt import App

# TODO: rename .env.example

# NEEDED Environment Variables 
openai_api_key = os.environ['OPENAI_API_KEY']
slack_app_token = os.environ['SLACK_BOT_APP_TOKEN']
slack_bot_token = os.environ['SLACK_API_KEY']
# Optional Environment variables
openai_engine = os.environ.get('OPENAI_ENGINE', 'gpt-3.5-turbo')
openai_max_tokens = int(os.environ.get('OPENAI_MAX_TOKENS', '1024'))
openai_ack_msg = os.environ.get('OPENAI_ACK_MSG', "Hello from your bot! :robot_face: \nThanks for your request, I'm on it!")
openai_reply_msg = os.environ.get('OPENAI_REPLY_MSG', "Here you go: \n")


# Event API & Web API
app = App(token=slack_bot_token) 
client = WebClient(slack_bot_token)

# This gets activated when the bot is tagged in a channel    
@app.event("app_mention")
def handle_message_events(body, logger):
    # Log message
    print(str(body["event"]["text"]).split(">")[1])
    
    # Create prompt for ChatGPT
    prompt = str(body["event"]["text"]).split(">")[1]
    
    # Let thre user know that we are busy with the request 
    response = client.chat_postMessage(channel=body["event"]["channel"], 
                                       thread_ts=body["event"]["event_ts"],
                                       text=f"{openai_ack_msg}")
    
    # Check ChatGPT
    openai.api_key = openai_api_key

    if openai_engine.startswith('gpt-3.'):
        response = openai.ChatCompletion.create(
            model=openai_engine,
            max_tokens=openai_max_tokens,
            stop=None,
            temperature=0.5,
            messages=[
                {"role": "user", "content": f"{prompt}"},
            ]
        ).choices[0].message.content

    else:
        response = openai.Completion.create(
            engine=openai_engine,
            prompt=prompt,
            max_tokens=openai_max_tokens,
            n=1,
            stop=None,
            temperature=0.5).choices[0].text
    
    
    # Reply to thread 
    response = client.chat_postMessage(channel=body["event"]["channel"], 
                                       thread_ts=body["event"]["event_ts"],
                                       text=f"{openai_reply_msg}{response}")

if __name__ == "__main__":
    SocketModeHandler(app, slack_app_token).start()
