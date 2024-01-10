from memory import get_chat_log, append_interaction_to_chat_log
import logging
import openai
import os
from gpt import *
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from pathlib import Path
from dotenv import load_dotenv


env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Set up your OpenAI API credentials
openai.api_key = os.environ["OPENAI_API_KEY"]

# Install the Slack app and get xoxb- token in advance
app = App(token=os.environ["SLACK_BOT_TOKEN"])

@app.command("/hello-socket-mode")
def hello_command(ack, body):
    user_id = body["user_id"]
    ack(f"Hi <@{user_id}>!")

#When the app is mentioned in a channel, respond with a message
@app.event("app_mention")
def event_test(event, say):
    #say(f"Hi there, <@{event['user']}>!")
    say(f"AI: ({ask(event['text'])})")
    #print(event['text'])
    #say(text=ask(text))

@app.command("/gen_image")
def handle_some_command(ack, body, logger):
    ack()
    logger.info(body)

#When the app is DM'd, respond with a message
@app.event("message")
def handle_message_events(event, say):
    #logger.info(body)
    #say(f"{ask_chatgpt(event['text'])}")

    #incoming_msg = request.values['Body']
    #chat_log = session.get('chat_log')
    #answer = ask(incoming_msg, chat_log)
    #session['chat_log'] = append_interaction_to_chat_log(incoming_msg, answer,
    #                                                     chat_log)

    #1. Getting data from incoming message
    time_stamp = event['event_ts']
    incoming_text = event['text']
    user = event['user']
    channel = event['channel']

    #2. Getting chat log from database to memory
    chat_log = get_chat_log("message", time_stamp, incoming_text, user, channel)

    print("BEFORE chat_log:- ", chat_log)
    #3. Getting answer from GPT
    answer = ask_chatgpt(incoming_text, chat_log)
    say(f"{answer}")

    #4. Updating chat log in database
    chat_log = append_interaction_to_chat_log(incoming_text, answer, chat_log, user, channel)
    print("AFTER chat_log:- ", chat_log)

def ack_shortcut(ack):
    ack()

def open_modal(body, client):
    client.views_open(
        trigger_id=body["trigger_id"],
        view={
            "type": "modal",
            "callback_id": "socket_modal_submission",
            "submit": {
                "type": "plain_text",
                "text": "Submit",
            },
            "close": {
                "type": "plain_text",
                "text": "Cancel",
            },
            "title": {
                "type": "plain_text",
                "text": "Socket Modal",
            },
            "blocks": [
                {
                    "type": "input",
                    "block_id": "q1",
                    "label": {
                        "type": "plain_text",
                        "text": "Write anything here!",
                    },
                    "element": {
                        "action_id": "feedback",
                        "type": "plain_text_input",
                    },
                },
                {
                    "type": "input",
                    "block_id": "q2",
                    "label": {
                        "type": "plain_text",
                        "text": "Can you tell us your favorites?",
                    },
                    "element": {
                        "type": "external_select",
                        "action_id": "favorite-animal",
                        "min_query_length": 0,
                        "placeholder": {
                            "type": "plain_text",
                            "text": "Select your favorites",
                        },
                    },
                },
            ],
        },
    )


app.shortcut("socket-mode")(ack=ack_shortcut, lazy=[open_modal])


all_options = [
    {
        "text": {"type": "plain_text", "text": ":cat: Cat"},
        "value": "cat",
    },
    {
        "text": {"type": "plain_text", "text": ":dog: Dog"},
        "value": "dog",
    },
    {
        "text": {"type": "plain_text", "text": ":bear: Bear"},
        "value": "bear",
    },
]

@app.options("favorite-animal")
def external_data_source_handler(ack, body):
    keyword = body.get("value")
    if keyword is not None and len(keyword) > 0:
        options = [o for o in all_options if keyword in o["text"]["text"]]
        ack(options=options)
    else:
        ack(options=all_options)

@app.view("socket_modal_submission")
def submission(ack):
    ack()

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
