import os
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request
import openai
import schedule
import time
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

app = Flask(__name__)
# Initializes your app with your bot token and signing secret
slack_app = App(
    token=os.environ.get('BOT_TOKEN'),
    signing_secret=os.environ.get("SINGING_SECRET"),
    raise_error_for_unhandled_request=True
)

slack_handler = SlackRequestHandler(slack_app)

# This will match any message that contains "Hello"
@slack_app.message("")
def answer(message, say):
    user = message['user']
    text = message['text']
    print(message['channel'])
    # say(f"Hello, glad to see you <@{user}>!")
    # say(channel=message['channel'],text=get_from_chatgpt(text))
    say(channel="C05JFRHTBPA" ,text=f"Hello, glad to see you <@{user}>!")


@app.route("/slack/events", methods=["POST"])
def slack_events():
    return slack_handler.handle(request)


def get_from_chatgpt(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: {prompt}\nAI:",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
    )
    response = response['choices'][0]['text']
    return response


# def send_message():
#     channel_id = "C05H8R9M8JX"
#     message = ":wave: Team, let's celebrate our daily wins! Every win counts, whether it's finishing a project, learning a skill, or making progress towards our goals. Share your successes to stay motivated and inspired. Keep up the great work!"
#     slack_app.client.chat_postMessage(channel=channel_id, text=message)
    

# schedule.every().day.at("18:01").do(send_message)

# Start your app
if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)))
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60) 
