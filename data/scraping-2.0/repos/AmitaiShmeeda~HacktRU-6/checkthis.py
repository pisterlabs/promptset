SLACK_BOT_TOKEN = "xoxb-5245319467909-5245510607157-DWgT5kU53gIceAp0A7hUyXD3"
SLACK_APP_TOKEN = "xapp-1-A057A9SCDA6-5251388510835-9585ef2b7c1ddae750bb39ca057e6a0c6d982adf158097d9b151e458880fe4e2"
OPENAI_API_KEY = "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"

import os
import openai
import slack_sdk
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack import WebClient
from slack_bolt import App

# Event API & Web API
app = App(token=SLACK_BOT_TOKEN)
client = WebClient(SLACK_BOT_TOKEN)


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
                                       text=f"Hello from your bot! :robot_face: \nThanks for your request, I'm on it!")

    # Check ChatGPT
    openai.api_key = OPENAI_API_KEY
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5).choices[0].text

    # Reply to thread
    response = client.chat_postMessage(channel=body["event"]["channel"],
                                       thread_ts=body["event"]["event_ts"],
                                       text=f"Here you go: \n{response}")


if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()