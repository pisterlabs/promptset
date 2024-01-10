import openai
import random
import os

MODEL = "gpt-3.5-turbo"

############ MOTIVATIONAL QUOTES ############
def getMotivationalQuote(topic):
    Motivational_Quote = openai.ChatCompletion.create(
        model=MODEL, 
        messages=[
            {
                "role": "user", 
                "content": f"Give me 10 motivational quotes with no repeats from people who are really good in the area of {topic_area} with no extra text",
            },
        ]
    )

    mquotes = Motivational_Quote['choices'][0]['message']['content']
    mquotes = mquotes.split('\n')
    mquotes = [mquote[3:] for mquote in mquotes]
    mquote = random.choice(mquotes)
    # hi


    return "You are doing great!"

def getPositiveMessage():
    return "You are awesome!"

def getFunFact():
    return "Did you know that the earth is round?"

def getMeanQuote():
    return "You are a failure!"

def processTestMessage(user, topic, messageType):
    if messageType == 'MQ':
        message = getMotivationalQuote(topic)
    elif messageType == 'SP':
        message = getPositiveMessage()
    elif messageType == 'FF':
        message = getFunFact()
    elif messageType == 'MD':
       message = getMeanQuote()
    else:
        message = 'Invalid message type'
    return message