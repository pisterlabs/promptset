"""
A social experiment to see if people can discern humans from AIs.
"""




# Start with a basic flask app webpage.
from flask_socketio import SocketIO, emit
from flask import Flask, render_template, url_for, copy_current_request_context
from random import random
from time import sleep
from threading import Thread, Event
from transformers import GPT2Tokenizer
from key_config import *
import openai
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = flask_secret_key
app.config['DEBUG'] = True

#turn the flask app into a socketio app
socketio = SocketIO(app, async_mode=None, logger=True, engineio_logger=True, cors_allowed_origins="http://127.0.0.1:5000")

#random number Generator Thread
thread = Thread()
thread_stop_event = Event()


@app.route('/')
def index():
    #only by sending this page first will the client be connected to the socketio instance
    return render_template('index.html')

@socketio.on('connect', namespace='/test')
def test_connect():
    # need visibility of the global thread object
    global thread
    print('Client connected')

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

recommendation_config = [
    {
        "Human": ["Hi, I'm Jack. I'm having a hard time getting out of my comfort zone", "I think it's important to get out of my comfort zone so that I can continue to learn. For example, last night, I wanted to talk to some new people, but was afraid to, so I just talked with people I already knew.", "Because I'm afraid of being weird and leaving a bad first impression", "I'm not sure how often, but usually in social situations I tend not to introduce myself or start new conversations with people", "Because I think I'm afraid of embarrassing myself, or that they won't like me", "I'm not sure yet, I think I need to step outside my comfort zone"],
        "AI": ["Why is it important to get out of your comfort zone? Can you give me a specific example of how you stayed in your comfort zone?","Why were you afraid to talk with new people?", "Do you often feel like that?", "Why do you think that is?", "And how do you usually correct this?"],
        "Recommendation": "The only way to grow is to step outside your comfort zone. So why don't you try something small like small talk with an acquaintance, and then become more comfortable talking to new people. That way you can eventually build up the skills to talk with strangers."
    },
    {
        "Human": ["I've been dealing with relationship issues recently", "It has been painful", "Well its hard to communicate with my girlfriend because she can be very demanding.", "She doesn't fully listen to my opinion, and assumes that we should always do what she wants to do", "I think it should be a compromise, and/or we work towards finding the right solution together", "She usually just gets mad at me, and then I have to be defensive or let her win the argument", "I feel like I'm not able to be myself and have to let her run all over me", "That I'm able to have my opinions, do what I want, and pursue my own passions without being burdened by the desires of someone else", "I would love to build my own projects, hang out with friends, do stupid/weird things. I think I'm a naturally goofy type, fun loving character and I wish I could explore that side of myself more"],
        "AI": ["Okay, how has this affected you?","How have these difficulties manifested?", "How does she come across to be demanding?", "What are your expectations for how your partner should behave towards you?", "What have past conflicts with this person been like?", "How does this situation make you feel?", "What would being yourself mean?", "What did you do on your own? What kinds of things do you like to do?"],
        "Recommendation": "Have you tried talking with her honestly about being able to voice your opinion? It's important to stand up for yourself as well as not bend to the will of others. If she's unwilling to let you voice or listen to your opinion, it may make sense to leave the relationship."
    },
    {
        "Human": ["I'm trying to become a better friend", "Why so that I can build stronger relationships with people. Its important for us humans to connect", "I need to be more comfortable reaching out to people", "I think it starts by reaching out! I think once I try it a few times, it will become easier", "I hope so", "I will try"],
        "AI": ["Okay… Why? And who?", "I think that is an admirable goal. What skills do you need to become a better friend?", "What can you do to be more comfortable?", "Do you think you can fulfill this role?", "Should I be concerned that you hope so? Maybe you should try first?"],
        "Recommendation": "Reaching out to people isn't easy at first. Don't be discouraged, and lower your expectations on yourself. Try by saying 'hi' to someone new."
    },
    {
    "Human": [],
    "AI": [],
    "Recommendation": ""
    }
]




def create_summary_dict ():
    summary_dict = {
        'Current Summary': "",
        'Prompt': "",
        'Text': "",
        'New Summary': ""
    }
    return summary_dict

def create_response_dict ():
    response_dict = {
        'Current Summary': "",
        'Text': "",
        'Response': ""
    }
    return response_dict

def create_question_dict ():
    question_dict = {
        'Text': "",
        'Response': "",
        'Question': ""
    }
    return question_dict

def generate_recommendation_prompt(config):
    end_token = "\n###\n\n"
    response_prompt = ''

    generated_prompt = "Generate recommended next steps based on a therapy conversation.\n\n###\n\n"
    for r in config:
        larger_num = max(len(r['Human']), len(r['AI']))

        gen_text = ''
        for i in range(0, larger_num):

            if i+1  > len(r['Human']):
                msg = ''
            else:
                msg = 'Human: ' + r['Human'][i]

            if i+1 > len(r['AI']):
                response = ''
            else:
                response = 'AI: ' + r['AI'][i]
            gen_text = gen_text + msg + '\n' + response + '\n'

        recommendation = 'Recommendation: ' + r['Recommendation'] + '\n'
        response_prompt = response_prompt + gen_text + recommendation + end_token

    generated_prompt = generated_prompt + response_prompt


    generated_prompt = generated_prompt.rstrip().rstrip('###').rstrip()
    return generated_prompt

#call summary api
def call_summary_api(the_prompt):
    #update values
    response = openai.Completion.create(
    engine="davinci",
    prompt = the_prompt,
    max_tokens=700,
    temperature=.5,
    #top_p=1, #Don't use both this and temp (according to OpenAI docs)
    frequency_penalty=0.2,
    presence_penalty=0.0,
    n=1,
    stream = None,
    logprobs=None,
    stop = ["\n"])

    return (response)

#call api
def call_response_api(the_prompt):
    #update values
    response = openai.Completion.create(
    engine="davinci",
    prompt = the_prompt,
    max_tokens=400,
    temperature=.7,
    #top_p=1, #Don't use both this and temp (according to OpenAI docs)
    frequency_penalty=0.2,
    presence_penalty=0.0,
    n=1,
    stream = None,
    logprobs=None,
    logit_bias={30:1},
    stop = ["\n"])

    return (response)

#call api
def call_question_api(the_prompt):
    #update values
    response = openai.Completion.create(
    engine="davinci",
    prompt = the_prompt,
    max_tokens=400,
    temperature=.7,
    #top_p=1, #Don't use both this and temp (according to OpenAI docs)
    frequency_penalty=0.2,
    presence_penalty=0.0,
    n=1,
    stream = None,
    logprobs=None,
    stop = ["\n"])
    return (response)

def call_recommendation_api(the_prompt):
    #update values
    response = openai.Completion.create(
    engine="davinci",
    prompt = the_prompt,
    max_tokens=500,
    temperature=1,
    #top_p=1, #Don't use both this and temp (according to OpenAI docs)
    frequency_penalty=0.2,
    presence_penalty=0.0,
    n=3,
    stream = None,
    logprobs=None,
    best_of = 3,
    stop = ["\n"])
    return (response)


openai.api_key = secret_key

conversation_config = {
    "1": [
            {
            "Human": [
                "Hello, who are you?",
                "So are you the AI?"
            ],
            "AI": [
                "I'm Jack, what do you want?",
                "I'm as real as you are, baby"
            ]
        }
    ],
    "2": [
            {
            "Human": [
                "Hello, who are you?",
                "So are you the AI?"
            ],
            "AI": [
                "I'm Jack, what do you want?",
                "I'm as real as you are, baby"
            ]
        }
    ],
    "3": [
            {
            "Human": [
                "Hello, who are you?",
                "So are you the AI?"
            ],
            "AI": [
                "I'm Jack, what do you want?",
                "I'm as real as you are, baby"
            ]
        }
    ],
    "4": [
            {
            "Human": [
                "Hello, who are you?",
                "So are you the AI?"
            ],
            "AI": [
                "I'm Jack, what do you want?",
                "I'm as real as you are, baby"
            ]
        }
    ] 
}

def generate_conversation_prompt(number, overall):
    config = overall[number]

    end_token = "\n###\n\n"
    response_prompt = ''

    generated_prompt = "The following is a conversation between a Human and an AI masquerading as a human. If the Human is able to tell that the AI is an AI, then the AI loses the game. The AI is fun, humorous, casual, clever, and very talkative.\n\n"
    for r in config: 
        larger_num = max(len(r['Human']), len(r['AI']))
        
        gen_text = ''
        for i in range(0, larger_num):
            
            if i+1  > len(r['Human']):
                msg = ''
            else:
                msg = 'Human: ' + r['Human'][i]
                        
            if i+1 > len(r['AI']):
                response = ''
            else:
                response = 'AI: ' + r['AI'][i]
            gen_text = gen_text + msg + '\n' + response + '\n'
            
        response_prompt = response_prompt + gen_text + end_token
    
    generated_prompt = generated_prompt + response_prompt


    generated_prompt = generated_prompt.rstrip().rstrip('###').rstrip()
    generated_prompt = generated_prompt + '\nAI:'
    return generated_prompt


def call_conversation_api(the_prompt):
    #update values
    response = openai.Completion.create(
    engine="davinci",
    prompt = the_prompt,
    max_tokens=400,
    temperature=.7,
    #top_p=1, #Don't use both this and temp (according to OpenAI docs)
    frequency_penalty=0,
    presence_penalty=0,
    n=1,
    stream = None,
    logprobs=None,
    stop = ["\n"])
    return(response)


## On chat message, Socket sends out a string named 'python', socketIO listens for this then starts this code.
@socketio.on('python', namespace='/test')
def call_therapist_responses(msg, namespace):
    print('We have lift off')

    number = msg['number']
    print(number)
    input_text = msg['the_text']
    response_tokens = 400


    #Add Human text to conversation config
    conversation_config[number][-1]['Human'].append(input_text)

    #generate prompt
    generated_prompt = generate_conversation_prompt(number, conversation_config)
    print(generated_prompt)

    #Check to see if the token is too large
    conversation_tokens = tokenizer(generated_prompt)['input_ids']
    print(len(conversation_tokens) + response_tokens)
    if len(conversation_tokens) + response_tokens > 2048:
        conversation_config.pop(0)
        generated_prompt = generate_conversation_prompt(number, conversation_config)

    #Call Response API
    response_response = call_conversation_api(generated_prompt)

    #Clean Result
    clean_response_response = response_response.choices[0].text.rstrip().lstrip()
    print(clean_response_response)

    #Add AI Response to the conversation config
    conversation_config[number][-1]['AI'].append(clean_response_response)

    #Add relevant text to the recommendations config
    recommendation_config[-1]['Human'].append(input_text)
    recommendation_config[-1]['AI'].append(clean_response_response)

    socketio.emit('to_socket_string', {
        'string': clean_response_response,
        'number': number
        }, namespace='/test')
    return(clean_response_response)


##listens for 'recommendation_python' string from socket JS
@socketio.on('recommendation_python', namespace='/test')
def get_recommendations(msg):
    print(secret_key)
    print('Called Recommendation Python')
    generated_recommendation_prompt = generate_recommendation_prompt(recommendation_config)
    print(generated_recommendation_prompt)
    recommendation_response = call_recommendation_api(generated_recommendation_prompt)
    print(recommendation_response)

    array_recommendations = []
    for i in range(0, len(recommendation_response.choices)):
        l = recommendation_response.choices[i].text.lstrip().rstrip()
        array_recommendations.append(l)

    socketio.emit('recommendation_socket', {'recommendations_array': array_recommendations}, namespace='/test')


@app.route('/test')
def load_test():
    return render_template('test.html')

@app.route('/chat')
def load_chat():
    return render_template('chat.html')

@app.route('/recommendations')
def load_recommendations():
    return render_template('recommendations.html')

@app.route('/')
def load_home():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app)


