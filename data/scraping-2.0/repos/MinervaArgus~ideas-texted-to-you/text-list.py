from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import time
import schedule
from twilio.rest import Client
import json

def generate_ideas():
    llm = Ollama(
        model="mistral-openorca", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    return (llm("Generate me a list of 5 ideas for a genius, unthought of, SaaS that i can easily make"))

def send_sms(ideas):
    account_sid = 'YourTwilioAccountSID'
    auth_token = 'YourTwilioAuthToken'
    client = Client(account_sid, auth_token)

    prefix = "Here are your SaaS ideas:\n"
    max_length = 1600 - len(prefix)  # Adjust for the length of the static message part

    # Split the ideas string into a list of individual ideas if necessary
    if isinstance(ideas, str):
        ideas = ideas.split(', ')  # Adjust this split based on how your ideas are separated

    messages = []
    current_message = prefix

    for idea in ideas:
        if len(current_message) + len(idea) + 2 > max_length:
            messages.append(current_message)
            current_message = prefix + idea
        else:
            if current_message != prefix:
                current_message += ', '
            current_message += idea

    if current_message != prefix:
        messages.append(current_message)

    # Send each message
    for msg in messages:
        message = client.messages.create(
            body=msg,
            from_='YourTwilioPhoneNum',
            to='YourPhoneNum'
        )

    return message.sid

def job():
    ideas = generate_ideas()
    response = send_sms(ideas)
    print(response)

# Schedule the job every 3 hours
schedule.every(3).hours.do(job)

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(1)

job()
