import openai_secret_manager

assert "openai" in openai_secret_manager.get_services()
secrets = openai_secret_manager.get_secret("openai")

import openai
openai.api_key = secrets["api_key"]

def get_tech_news():
    response = openai.Completion.create(
        engine="davinci",
        prompt="Give me a list of the top 5 tech news stories for today with a brief explanation of each.",
        n=1,
        max_tokens=150,
        temperature=0.5
    )
    return response['choices'][0]['text']

def handle_message(message):
    if message == 'tech news':
        news = get_tech_news()
        return news
    else:
        return "Sorry, I didn't understand that. Please type 'tech news' to get a list of the top 5 tech news stories for today with a brief explanation of each."

def handle_command(command):
    if command["type"] == "message":
        return handle_message(command["data"]["text"])
    else:
        return "Sorry, I didn't understand that command."

def process_event(event):
    if event["type"] == "message":
        response_text = handle_message(event["data"]["text"])
        openai.Completion.create(
            engine="davinci",
            prompt=response_text,
            n=1,
            max_tokens=150,
            temperature=0.5
        )
