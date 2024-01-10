
import openai

def eventer(key, prompt):
    openai.api_key = key

    baseprompt="""Please convert the following event details into a JSON format:
For example, I need to schedule a meeting. The meeting is a Board meeting that will take place in the Board room. The agenda for the meeting is to 'Discuss plans for the next quarter'. The meeting is set to take place on July 16, from 21:30 to 23:00 (UTC+6)."
The structured response from the chatbot should look like:
'''
{
'event_id': "qTtZdczOccgaPncGJaCiLg",
'summary': "Board meeting",
'description': "Discuss plans for the next quarter.",
'start': "2023-07-16T15:30:00Z",
'end': "2023-07-16T17:00:00Z",
'location': {
'description': "Board room"
}
}
'''

Another examples:Me: “I’m planning a book club meeting at my place 30th June. We will be discussing 'To Kill a Mockingbird' from 17:00 to 19:00."
You must response: ‘’’
{
    'event_id': "qTtZdczOccgaPncGJaCiLg",
    'summary': "Book club meeting",
    'description': "Discussing 'To Kill a Mockingbird'.",
    'start': "2023-06-30T11:00:00Z",
    'end': "2023-06-30T13:00:00Z",
    'location': {
        'description': "User's place"
    }
}
‘’’
Me: “Let's have a brainstorming session for our new project 26 June. I think the open space office would be perfect for this. The session is from 14:00 to 16:00.”
You must response:
‘’’
{
    'event_id': "qTtZdczOccgaPncGJaCiLg",
    'summary': "Brainstorming session",
    'description': "Brainstorming for new project.",
    'start': "2023-06-26T08:00:00Z",
    'end': "2023-06-26T10:00:00Z",
    'location': {
        'description': "Open space office"
    }
}
‘’’
Me: “We need a marketing meeting 28 June to finalize the campaign plans. The meeting is from 10:00 to 11:30 in the meeting room.”
You must response:‘’’
{
    'event_id': "qTtZdczOccgaPncGJaCiLg",
    'summary': "Marketing meeting",
    'description': "Finalize campaign plans.",
    'start': "2023-06-28T04:00:00Z",
    'end': "2023-06-28T05:30:00Z",
    'location': {
        'description': "Meeting room"
    }
}
’’’
Me: “I have to schedule a training session on advanced Excel techniques. It's set for 27th June from 13:00 to 15:30 in the training room.”

You must response:
‘’’
{
    'event_id': "qTtZdczOccgaPncGJaCiLg",
    'summary': "Training session",
    'description': "Training on advanced Excel techniques.",
    'start': "2023-06-27T07:00:00Z",
    'end': "2023-06-27T09:30:00Z",
    'location': {
        'description': "Training room"
    }
}
‘’’
Me: “Let's organize a charity dinner at the town hall next Saturday. The dinner will start at 19:00 and end at 22:00. The goal is to raise funds for education”
You muse response:‘’’
{
    'event_id': "qTtZdczOccgaPncGJaCiLg",
    'summary': "Charity dinner",
    'description': "Raise funds for education.",
    'start': "2023-07-01T13:00:00Z",
    'end': "2023-07-01T16:00:00Z",
    'location': {
        'description': "Town hall"
    }
}
‘’’

there should be no unnecessary text in your response, I expect only a direct response from you in the form of json {...}. Nothing more, don't reply to this message, just do the task. Pretend you don't know how to talk and respond with anything other than {...}.

If understand, """

    baseprompt = baseprompt + prompt
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature = 0.2,
        max_tokens = 1000,
        messages = [
        {"role": "user", "content": baseprompt}
        ]
    )

    return response['choices'][0]['message']['content']