import os
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import openai

# Loaded environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Debug: Print the loaded environment variables
print("SLACK_BOT_TOKEN:", os.environ.get("SLACK_BOT_TOKEN"))

# Initialized the Slack app
app = App(token=os.environ["SLACK_BOT_TOKEN"])
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Function to generate an image using DALL·E API
def generate_dalle_image(prompt):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    return image_url

# Defined the bot's behavior when it receives a message
@app.event("app_mention")
def handle_app_mention_events(body, say):
    # Extracted the text after the mention (excluding the mention itself)
    text = body["event"]["text"].replace(app.client.token, "").strip()
    
    try:
        image_url = generate_dalle_image(text)
        response = say(f"Here's an image based on your input: {image_url}")
    except Exception as e:
        response = say(f"An error occurred: {str(e)}")
    
    return response

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()














