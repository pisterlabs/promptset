"""
Demo Flask application to test the operation of Flask with socket.io

Aim is to create a webpage that is constantly updated with random numbers from a background python process.

30th May 2014

===================

Updated 13th April 2018

+ Upgraded code to Python 3
+ Used Python3 SocketIO implementation
+ Updated CDN Javascript and CSS sources

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
socketio = SocketIO(app, async_mode=None, logger=True, engineio_logger=True)

#random number Generator Thread
thread = Thread()
thread_stop_event = Event()

def randomNumberGenerator():
    """
    Generate a random number every 1 second and emit to a socketio instance (broadcast)
    Ideally to be run in a separate thread?
    """
    #infinite loop of magical random numbers
    print("Making random numbers")
    while not thread_stop_event.isSet():
        number = round(random()*10, 3)
        print(number)
        socketio.emit('newnumber', {'number': number}, namespace='/test')
        socketio.sleep(5)


@app.route('/')
def index():
    #only by sending this page first will the client be connected to the socketio instance
    return render_template('index.html')

@socketio.on('connect', namespace='/test')
def test_connect():
    # need visibility of the global thread object
    global thread
    print('Client connected')

    #Start the random number generator thread only if the thread has not been started before.
    if not thread.isAlive():
        print("Starting Thread")
        thread = socketio.start_background_task(randomNumberGenerator)

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

summary_config = [
    {
        "Current Summary": "Cassie is in her first Therapy session. She is unsure if she should be here.",
        "Prompt": "Okay. Why don't we start by discussing what has been on your mind?",
        "Text": "Well, I am very frustrated lately. You see, I work as an executive assistant at SII, but it's really just a means to make money while I pursue my real goal--to be an actress. I have been taking classes since I was, like, 8 or 9 years old. Hollywood has always fascinated me. Can you imagine living in those times? The golden age of the cinema. Fred and Ginger waltzing across the big screen, Vivien Leigh and 'Gone With The Wind', Casablanca.",
        "New Summary": "Cassie is in her first Therapy session and is unsure if she should be here. Cassie has been frustrated lately. She works as an executive assistant at SII. She has always wanted to be an actress."
    },
    {
        "Current Summary": "First Message",
        "Prompt": "What is your name, and what is on your mind?",
        "Text": "Hi, it's nice to meet you. I've been feeling like I've been losing control of my life lately. I've been trying to balance being great at my job with pursuing my passions, but I feel like I don't have enough time in the week.",
        "New Summary": "User feels like they are losing control of their life. They want to pursue their passions, but they don't feel like they have enough time."
    },
    {
        "Current Summary": "User feels like they've been in a rut lately because they didn't get much work done today.",
        "Prompt": "Why do you think you didn't get much work done today?",
        "Text": "I think its because I didn't go into the day with a plan. Like I didn't come into the day with goals I'd like to accomplish.",
        "New Summary": "User feels like they've been in a rut lately because they didn't get much work done today. This is because they didn't come into the day with a plan or goals."
    },
    {
        "Current Summary": "Jack feels like he's losing control of his life. He wants to pursue his passions but he doesn't feel like he has enough time. ",
        "Prompt": "I understand. What are your passions?",
        "Text": "Well, I want to start software companies in Silicon Valley. It's my dream to be a CEO, but right now I don't have the requisite skills to do so. Like I need to be able to code better, I need to be able to communicate better, and I need more courage. It's tough because I want to learn and grow and build these skills, but I also want people to respect me at my job. I just want to take a big risk sometimes.",
        "New Summary": "Jack feels like he is losing control of his life because he doesn't have the time to pursue his passion of starting a company. He doesn't think he currently have the skills to achieve his dreams and doesn't feel like he has enough time to learn them."
    },
    {
        "Current Summary": "Cassie has been frustrated lately. She works as an executive assistant at SII. She has always wanted to be an actress. She is in an improvisational troupe that she enjoys, but it's hard on her because her boyfriend doesn't like that she's acting. He wants her to build a real career and be a productive bread winner, and get more serious about her relationship. She thinks he is jealous because she has less time for him, and because sometimes she does love scenes in the improv troupe.",
        "Prompt": "How is your relationship with him besides the acting?",
        "Text": "Well...good, I guess. We met in college, and have been dating ever since then. I do really like him, maybe even love him, but...he can be kind of demanding at times.",
        "New Summary": "Cassie has been frustrated lately. She works as an executive assistant at SII. She has always wanted to be an actress. She is in an improvisational troupe that she enjoys, but it's hard on her because her boyfriend doesn't like that she's acting. He wants her to build a real career and be a productive bread winner, and get more serious about her relationship. She thinks he is jealous because she has less time for him, and because sometimes she does love scenes in the improv troupe. Her relationship with him is good, but he can be demanding at times."
    },
    {
        "Current Summary": "First Message",
        "Prompt": "Hi, I'm an AI Therapist. What's on your mind?",
        "Text": "",
        "New Summary": ""
    }
]

response_config = [
    {
        'Current Summary': "User has a gambling problem. They took the highway on the way home, passed a casino, and went to it. She thought that she could just go for a few hours like a regular person, but ended up staying the whole night. She's in serious gambling debt, and lost a lot more money that night. Her partner also knows about her relapse.",
        'Text': "Well, I was thinking that I hadn't gambled in a month and it'd been pretty easy to stay away for that long so I thought maybe I don't have a problem. So I wanted togamble like a regular person, you know, just go in for a bit and then leave.",
        'Response': "So you were wanting to test out personal control as well as test whether or not you really have a gambling problem?"
    },
    {
        'Current Summary': "User is an Afghanistan War Veteran. They are suffering from PTSD from an improvised explosive device that killed their fellow servicemen. They feel guilty about their death because they waved their truck on ahead, after another truck broke. Shortly after being waved ahead, that truck hit the Improvised Explosive Device. She feels at fault for their death.",
        'Text': "If I hadn’t have waved them through and told them to carry on, this wouldn’t have happened. It is my fault that they died. (Begins to cry)",
        'Response': "It is certainly sad that they died. However, I want us to think through the idea that you should have had them wait and not had them go on, and consequently that it was your fault. If you think back about what you knew at the time — not what you know now 5 years after the outcome — did you see anything that looked like a possible explosive device when you were scanning the road as the original lead truck?"
    },
    {
        'Current Summary': "Jack feels like he's losing control of his life. He wants to pursue his passions but he doesn't feel like he has enough time.",
        'Text': "Hi, I'm Jack. It's nice to meet you. I've been feeling like I've been losing control of my life lately. I've been trying to balance being great at my job with pursuing my passions, but I feel like I don't have enough time in the week.",
        'Response': "I think I understand. You mentioned pursuing your passions... What are they?"
    },
    {
        'Current Summary': "Ms. Evans, also known as Cassie, is in her first Therapy session. Cassie has been frustrated lately. She works as an executive assistant at SII. She has always wanted to be an actress. She is in an improvisational troupe that she enjoys, but it's hard on her because her boyfriend doesn't like that she's acting. He thinks she should build a real career and be a productive bread winner. She thinks he is jealous.",
        'Text': "Well to start off, my boyfriend is not very happy with me and the acting thing. He thinks it's ridiculous because it is so hard to make a living at it. And in the meantime, I could be building a 'real' career. He wants to get more serious with me, but wants to make sure that I can be a productive bread winner. And I think he is jealous.",
        'Response': "Jealous? How so?"
    },
    {
        'Current Summary': "Jack feels like he is losing control of his life because he doesn't have the time to pursue his passion of starting a company. He doesn't think he currently have the skills to achieve his dreams and doesn't feel like he has enough time to learn them.",
        'Text': "Well, I want to start software companies in Silicon Valley. It's my dream to be a CEO, but right now I don't have the requisite skills to do so. Like I need to be able to code better, I need to be able to communicate better, and I need more courage. It's tough because I want to learn and grow and build these skills, but I also want people to respect me at my job. I just want to take a big risk sometimes.",
        'Response': "Sometimes it can be hard to balance everything in your life. Why do you want to be a CEO for a software company?"
    }
]

question_config = [
    {
        'Text': "Hi, I'm Derek... I've been going through a rough time lately with my girlfriend. She wants so much of my time and she can be demanding. It feels like she is controlling my life and not me. She's always planning things so far ahead, and I feel bad by saying no because they feel fun. But what ends up happening is that I don't actually do what I want to do by the time the plans come to fruition... Sometimes I'm afraid to say no.",
        'Response': "I want you to understand, and let me emphasize this... Sometimes it's okay to say no.",
        'Question': "How is your relationship with your girlfriend?"
    },
    {
        'Text': "We were up late talking and then crashed on the living room floor. It was sweet and nice. And he didn't make any stupid comments about having sex. Or not having it, rather. It was the first time in a long while that we didn't argue a lot; and we actually talked about stuff. Meaningful things. It was like when we first started going out.",
        'Response': "I'm glad you're starting to open up to your boyfriend.",
        'Question': "How are you feeling right now?"
    },
    {
        'Text': "Well, I want to start software companies in Silicon Valley. It's my dream to be a CEO, but right now I don't have the requisite skills to do so. Like I need to be able to code better, I need to be able to communicate better, and I need more courage. It's tough because I want to learn and grow and build these skills, but I also want people to respect me at my job. I just want to take a big risk sometimes.",
        'Response': "Sometimes it can be hard to balance everything in your life.",
        'Question': "Have you ever thought that you're putting too much pressure on yourself?"
    },
    {
        'Text': "My boss thinks I'm doing great at work, but my peers aren't respecting my work. I spent an entire month working on one review and the information wasn't even used. I just feel like I am not being respected or looked up to.",
        'Response': "At the end of the day, you can only focus on what you can control.",
        'Question': "Let's switch the topic. Overall, how's everything in your life right now?"
    }
]

recommendation_config = [
    {
        "User": ["Hi, I'm Jack. I'm having a hard time getting out of my comfort zone", "I think it's important to get out of my comfort zone so that I can continue to learn. For example, last night, I wanted to talk to some new people, but was afraid to, so I just talked with people I already knew.", "Because I'm afraid of being weird and leaving a bad first impression", "I'm not sure how often, but usually in social situations I tend not to introduce myself or start new conversations with people", "Because I think I'm afraid of embarrassing myself, or that they won't like me", "I'm not sure yet, I think I need to step outside my comfort zone"],
        "AI": ["Why is it important to get out of your comfort zone? Can you give me a specific example of how you stayed in your comfort zone?","Why were you afraid to talk with new people?", "Do you often feel like that?", "Why do you think that is?", "And how do you usually correct this?"],
        "Recommendation": "The only way to grow is to step outside your comfort zone. So why don't you try something small like small talk with an acquaintance, and then become more comfortable talking to new people. That way you can eventually build up the skills to talk with strangers."
    },
    {
        "User": ["I've been dealing with relationship issues recently", "It has been painful", "Well its hard to communicate with my girlfriend because she can be very demanding.", "She doesn't fully listen to my opinion, and assumes that we should always do what she wants to do", "I think it should be a compromise, and/or we work towards finding the right solution together", "She usually just gets mad at me, and then I have to be defensive or let her win the argument", "I feel like I'm not able to be myself and have to let her run all over me", "That I'm able to have my opinions, do what I want, and pursue my own passions without being burdened by the desires of someone else", "I would love to build my own projects, hang out with friends, do stupid/weird things. I think I'm a naturally goofy type, fun loving character and I wish I could explore that side of myself more"],
        "AI": ["Okay, how has this affected you?","How have these difficulties manifested?", "How does she come across to be demanding?", "What are your expectations for how your partner should behave towards you?", "What have past conflicts with this person been like?", "How does this situation make you feel?", "What would being yourself mean?", "What did you do on your own? What kinds of things do you like to do?"],
        "Recommendation": "Have you tried talking with her honestly about being able to voice your opinion? It's important to stand up for yourself as well as not bend to the will of others. If she's unwilling to let you voice or listen to your opinion, it may make sense to leave the relationship."
    },
    {
        "User": ["I'm trying to become a better friend", "Why so that I can build stronger relationships with people. Its important for us humans to connect", "I need to be more comfortable reaching out to people", "I think it starts by reaching out! I think once I try it a few times, it will become easier", "I hope so", "I will try"],
        "AI": ["Okay… Why? And who?", "I think that is an admirable goal. What skills do you need to become a better friend?", "What can you do to be more comfortable?", "Do you think you can fulfill this role?", "Should I be concerned that you hope so? Maybe you should try first?"],
        "Recommendation": "Reaching out to people isn't easy at first. Don't be discouraged, and lower your expectations on yourself. Try by saying 'hi' to someone new."
    },
    {
    "User": [],
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

def generate_summary_prompt(config):
    end_token = "\n###\n\n"

    generated_prompt = "A summarizing tool for therapy.\n\n###\n\n"
    for r in config:
        current_summary = "Current Summary: " + r['Current Summary'] + "\n"
        prompt = "Prompt: " + r['Prompt'] + "\n"
        text = "Text: " + r['Text'] + "\n"
        new_summary = "New Summary: " + r['New Summary'] + "\n"

        response_prompt = current_summary + prompt + text + new_summary + end_token
        generated_prompt = generated_prompt + response_prompt


    generated_prompt = generated_prompt.rstrip().rstrip('###').rstrip()
    return generated_prompt

def generate_response_prompt(config):
    end_token = "\n###\n\n"

    generated_prompt = "The following is a conversation with an AI therapist. The therapist is helpful, creative, clever, and very friendly. At the beginning of the conversation, they ask questions to learn more about the User's problems, but once the Current Summary gets long, they start to offer recommendations.\n\n"
    for r in config:
        current_summary = "Current Summary: " + r['Current Summary'] + "\n"
        text = "Text: " + r['Text'] + "\n"
        response = "Response: " + r['Response'] + "\n"

        response_prompt = current_summary + text + response + end_token
        generated_prompt = generated_prompt + response_prompt


    generated_prompt = generated_prompt.rstrip().rstrip('###').rstrip()
    return generated_prompt

def generate_question_prompt(config):
    end_token = "\n###\n\n"

    generated_prompt = "Therapist asking a question based on the past response and the previous users text. The goal is for the question to help the user open up about their problems.\n\n"
    for r in config:
        text = "Text: " + r['Text'] + "\n"
        response = "Response: " + r['Response'] + "\n"
        question = "Question: " + r['Question'] + "\n"

        response_prompt = text + response + question + end_token
        generated_prompt = generated_prompt + response_prompt


    generated_prompt = generated_prompt.rstrip().rstrip('###').rstrip()
    return generated_prompt

def generate_recommendation_prompt(config):
    end_token = "\n###\n\n"
    response_prompt = ''

    generated_prompt = "Generate recommended next steps based on a therapy conversation.\n\n###\n\n"
    for r in config:
        larger_num = max(len(r['User']), len(r['AI']))

        gen_text = ''
        for i in range(0, larger_num):

            if i+1  > len(r['User']):
                msg = ''
            else:
                msg = 'User: ' + r['User'][i]

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

@socketio.on('python_old', namespace='/test')
def call_therapist(msg, namespace):
    print(msg['the_text'])
    input_text = msg['the_text']

    #input text into the generated starter dialogue
    summary_config[-1]['Text'] = input_text
    #generate prompt
    generated_summary_prompt = generate_summary_prompt(summary_config)

    #Check to see if the token is too large
    summary_tokens = tokenizer(generated_summary_prompt)['input_ids']
    print(len(summary_tokens) + 700)
    if len(summary_tokens) + 700 > 2048:
        summary_config.pop(1)
        generated_summary_prompt = generate_summary_prompt(summary_config)

    #Call Summary API
    summary_response = call_summary_api(generated_summary_prompt)
    #Clean Result
    clean_summary_response = summary_response.choices[0].text.rstrip().lstrip()
    #Add New Summary to Last Summary Config
    summary_config[-1]['New Summary'] = clean_summary_response

    #Create new response config and add requisite info
    new_response_config = create_response_dict()
    new_response_config['Text'] = input_text
    new_response_config['Current Summary'] = clean_summary_response

    #Create new summary config and add summary response
    new_summary_config = create_summary_dict()
    new_summary_config['Current Summary'] = clean_summary_response

    #Add new response config to the old response config
    response_config.append(new_response_config)
    #Generate prompt for response config and call API
    generated_response_prompt = generate_response_prompt(response_config)

    #Check to see if response prompt is too large
    response_tokens = tokenizer(generated_response_prompt)['input_ids']
    print(len(response_tokens) + 400)
    if len(response_tokens) + 400 > 2048:
        response_config.pop(1)
        generated_response_prompt = generate_summary_prompt(response_config)

    #Call Response API
    response_response = call_response_api(generated_response_prompt)
    #Clean Result
    clean_response_response = response_response.choices[0].text.rstrip().lstrip()
    #Add new response to new response config
    new_response_config['Response'] = clean_response_response

    #Add new response to new summary config
    new_summary_config['Prompt'] = clean_response_response
    print(len(clean_response_response))

    clean_question_response = ''
    #test if the response ended in a question. If not, call the question api
    if clean_response_response[len(clean_response_response) - 1] != '?':
        print('The forced question section!!!!')
        new_question_dict = create_question_dict()
        new_question_dict['Text'] = input_text
        new_question_dict['Response'] = clean_response_response

        question_config.append(new_question_dict)
        generated_question_prompt = generate_question_prompt(question_config)
        print(generated_question_prompt)

        question_response = call_question_api(generated_question_prompt)
        clean_question_response = question_response.choices[0].text.rstrip().lstrip()

        clean_response_response = clean_response_response + ' ' + clean_question_response

        question_config.pop(len(question_config) - 1)

    #Add both new configs to the old configs
    summary_config.append(new_summary_config)
    summary_config.pop(3)
    response_config.pop(2)

    #Add relevant text to the recommendations config
    recommendation_config[-1]['User'].append(input_text)
    recommendation_config[-1]['AI'].append(clean_response_response)


    print(generated_summary_prompt)
    print(clean_summary_response)
    socketio.emit('to_socket_string', {'string': clean_response_response}, namespace='/test')

conversation_config = [
    {
        "User": ["I've been feeling really sad over the loss of my boyfriend",
                 "Well, I keep felling like I just can’t move on like …I have this baggage that is going to follow me with whomever I meet or…or any path I go in life.",
                 "No.. I guess it doesn’t but……….I don’t know.",
                 "Well……I think it has something to do with my dad.",
                 "Well, I guess growing up my dad never made me feel like I was important…. always saying things to belittle me and stuff like that. And then…when…Eric would do that stuff to me, you know like, he would go riding and not pay any attention to me or…just play poker and…disregard whatever I said to him, it kind of made me feel the same way, you know?",
                 "Yeah…well…I guess it’s all my fault though, you know because I just can’t control my attitude",
                 "I guess if….I guess I can if I really tried.",
                 "Well, I don’t thinks so.",
                 "Yeah.",
                 "Umm…I..I just can’t control my mouth sometimes. Like I think that I just provoke them by just things that I say, you know. Maybe to them it stupidity, I don’t know, but, I guess I say things to hurt them and then it just instigates a fight automatically.",
                 "Mmmm….(silence) Like, that it hurts me?"],
        "AI": ["How are you handling the feelings of sadness?",
               "Does this baggage you’re referring to have to follow you around?",
               "Where do you think this baggage really comes from?",
               "I see…Talk to me about that.",
               "So you’re feeling like your boyfriend reminds you of when your father used to…..make you feel like you weren’t important",
               "So you can’t control you’re attitude huh",
               "Are you saying you haven’t really tried?",
               "Earlier you were talking about how your boyfriend reminds you of your dad…",
               "Tell me… How do you think that all ties into with your attitude?",
               "It sounds like this attitude your referring to may be a way to protect your feelings.",
               "Yeah, like that it hurts you.... You know, anger usually comes right after hurtful feelings and the attitude may be a way to cope with those feelings."
               ]
    },
    {
        "User": ["Its been tough lately. My parents have been dealing with marital issues, and I just did poorly on my exams. It's just annoying because I know that I should have done better during my exams, and I just didn't. I guess I just beat myself up a lot",
                 "Maybe more than other people. My parents have always told me in a loving way that you should always do your best and get the best. I agree with that, but when I don't do that... I don't know... it annoys me a lot, and I'm afraid it annoys them",
                 "No...",
                 "I think it doesn't really sound good. I don't really talk to them a lot. I don't try to talk to them because it makes me feel worse, but I don't think their relationship is getting any better.",
                 "For a little bit",
                 "Not very fun... No... it was quite tense most of the time. It stressed me out more than it relaxed me."],
        "AI": ["You say 'I know I should have done better.' I guess hearing it from perspective, I'm thinking you had a lot going on at the time. It'd be understandable absolutely that you wouldn't do as well as normal. I suppose I'm just wondering if you have very high standards for yourself.",
               "It sounds like you feel you should always do your best. Are there any kind of exceptions to that?",
               "Okay, so I can then that was a very difficult time for you. What's the situation with your parents at the moment?",
               "Were you home for the Summer?",
               "And how was that?",
               "Okay, it sounds like a lot has been happening. You've been feeling very low, you've had University pressure on, and you've had problems with your parents. You can't really feel like you can reach out to your parents. So I suppose then, could you tell me a bit more about how you've been feeling this past week?"
               ]
    },
    {
        "User": ["I got frustrated on Friday when I had just implemented a new policy for staff members. I had imagined that I would get a lot of phone calls about it because I always do but I ended up snapping at people over the phone.",
                 "I felt quite stressed and also annoyed at other staff members because they didn’t understand the policy.",
                 "I guess I was thinking that no-one appreciates what I do."],
        "AI": ["And how were you feeling at that time?",
               "And what was going through your mind?",
               "Okay. You just identified what we call an automatic thought. Everyone has them. They are thoughts that immediately pop to mind without any effort on your part. Most of the time the thought occurs so quickly you don’t notice it but it has an impact on your emotions. It’s usually the emotion that you notice, rather than the thought. Often these automatic thoughts are distorted in some way but we usually don’t stop to question the validity of the thought. Tell me, what is the effect of believing that ‘no-one appreciates you?’",]
    },
    {
        "User": [],
        "AI": []
    }
]

def generate_conversation_prompt(config):
    end_token = "\n###\n\n"
    response_prompt = ''

    generated_prompt = "The following is a conversation with an AI therapist. The AI is helpful, clever, and humorous. They want to offer recommendations to help the User deal with their issues.\n\n###\n\n"
    for r in config: 
        larger_num = max(len(r['User']), len(r['AI']))
        
        gen_text = ''
        for i in range(0, larger_num):
            
            if i+1  > len(r['User']):
                msg = ''
            else:
                msg = 'User: ' + r['User'][i]
                        
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

@socketio.on('python', namespace='/test')
def call_therapist_responses(msg, namespace):
    print('We have lift off')

    input_text = msg['the_text']
    response_tokens = 400


    #Add user text to conversation config
    conversation_config[-1]['User'].append(input_text)

    #generate prompt
    generated_prompt = generate_conversation_prompt(conversation_config)
    print(generated_prompt)

    #Check to see if the token is too large
    conversation_tokens = tokenizer(generated_prompt)['input_ids']
    print(len(conversation_tokens) + response_tokens)
    if len(conversation_tokens) + response_tokens > 2048:
        conversation_config.pop(0)
        generated_prompt = generate_conversation_prompt(conversation_config)

    #Call Response API
    response_response = call_conversation_api(generated_prompt)

    #Clean Result
    clean_response_response = response_response.choices[0].text.rstrip().lstrip()
    print(clean_response_response)

    #Add AI Response to the conversation config
    conversation_config[-1]['AI'].append(clean_response_response)

    #Add relevant text to the recommendations config
    recommendation_config[-1]['User'].append(input_text)
    recommendation_config[-1]['AI'].append(clean_response_response)

    socketio.emit('to_socket_string', {'string': clean_response_response}, namespace='/test')
    return(clean_response_response)

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


