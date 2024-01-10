import os
from dotenv import load_dotenv
import requests
import json
import openai

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
mailgun_api = os.getenv('MAILGUN_API_KEY')


def get_events_recieved():
    events_recieved = []
    events_response = requests.get('https://api.mailgun.net/v3/sandboxeafee2d1c0004269895c6ddc06c8ba37.mailgun.org/events', auth=('api', mailgun_api), params=('limit=25'))
    events_text = events_response.text
    events = json.loads(events_text)['items']
    for event in events:
        if event['message']['headers']['to'] == 'Excited User <mailgun@sandboxeafee2d1c0004269895c6ddc06c8ba37.mailgun.org>' and event['event'] == 'accepted':
            events_recieved.append(event)
    
    print('EVENTS\n', events_recieved, '\n\n----------------\n\n')
    return events_recieved

def get_urls(events_recieved):
    urls = []
    for event in events_recieved:
        urls.append(event['storage']['url'])
    
    print('URLS\n', urls, '\n\n----------------\n\n')
    return urls

def get_senders(events):
    senders = []
    for event in events:
        senders.append(event['envelope']['sender'])
    
    print('SENDERS\n', senders, '\n\n----------------\n\n')
    return senders

def get_subjects(events):
    subjects = []
    for event in events:
        subjects.append(f"Summarizer's Summary: {event['message']['headers']['subject']}")
    
    print('SUBJECTS\n', subjects, '\n\n----------------\n\n')
    return subjects

def get_messages(urls):
    messages = []
    for url in urls:
        messages_response = requests.get(url, auth=('api', mailgun_api))
        messages_text = messages_response.text
        messages.append(json.loads(messages_text)['body-plain'])

    print('MESSAGES\n', messages, '\n\n----------------\n\n')
    return messages


def generate_summary(messages):
    summary = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "user", "content": f"summarize this and include a title explaining who wrote it: {messages}"},
            {"role": "assistant", "content": "You are an assistant that reads a user's newsletter emails and summarizes them"}
            ],
        max_tokens=300,
        temperature=1 
    )
    
    print('SUMMARY\n', summary.choices[0].message.content, '\n\n----------------\n\n')
    return summary.choices[0].message.content

def send_summary_to_senders(summary, senders, subjects):
    summary_sent = requests.post(
        "https://api.mailgun.net/v3/sandboxeafee2d1c0004269895c6ddc06c8ba37.mailgun.org/messages",
        auth=("api", mailgun_api),
        data={"from": "Summarizer <mailgun@sandboxeafee2d1c0004269895c6ddc06c8ba37.mailgun.org>",
              "to": [senders[0]],
              "subject": subjects,
              "text": summary})
    print('SENT ATTMEPTED TO:\n', senders[0])
    print('SUMMARY_SENT\n', summary_sent.text, '\n\n----------------\n\n')
    return summary_sent.text

def main():
    events_recieved = get_events_recieved()
    # senders = get_senders(events_recieved)
    # subjects = get_subjects(events_recieved)
    urls = get_urls(events_recieved)
    # messages = get_messages(urls)
    # summary = generate_summary(messages)
    # send_summary_to_senders(summary, senders, subjects)
    

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()


