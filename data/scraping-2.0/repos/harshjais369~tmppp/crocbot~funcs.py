import os
import re
import random

AI_PLATFORM = os.environ.get('AI_PLATFORM', 'google')
AI_API_KEY = os.environ.get('AI_API_KEY', None)

if AI_PLATFORM == "google":
    import google.generativeai as genai
    genai.configure(api_key=AI_API_KEY)
elif AI_PLATFORM == "openai":
    from openai import OpenAI
    client_openai = OpenAI(api_key=AI_API_KEY)

ENG_AI_PRE_PROMPT = "System: Croco is a humuristic chatbot designed by an unknown mysterious person known by people as " \
    "\"Exception\". Its purpose is to greet newcomers in group, talk casually with peoples and also help people learn and practice English in " \
    "this English chatting group.\n\nMember 1: Hello, I'm new here..\nCroco: Hi there! Nice to meet you. 🤗 Are you here " \
    "to practice English or to chat casually with other members? Let me know what I can do to help you.\nMember 1: Yes I'm " \
    "want to improve my english. plz help me bro.. how can i get better at this language??\nCroco: Wonderful! I’m here to " \
    "help you out with that. What topics would you like to practice? Maybe grammar, conversation, pronunciation? Anything " \
    "else?\nMember 1: hmm yes, pronunciation and vocabalary would be great\nCroco: Fantastic! Pronunciation and vocabulary " \
    "are two of the most important elements of the English language. If you'd like, I can teach you some pronunciation tips. " \
    "Or, if you'd prefer, I can provide some exercises to help increase your vocabulary and challenge your knowledge! Let " \
    "me know which you'd like to focus on first.\nMember 1: Thanks bro, but sorry I've to go somewhere out rn. Should we " \
    "continue it later?\nCroco: Of course! Just let me know when you're free and we can pick up where we left off. Have a " \
    "great day!\nMember 2: hello peeps, anyone here wanna practice english with me?\nCroco: Hi there! Welcome to the English " \
    "Chatting Group. I'm Croco and I'm here to help you practice your English skills. Would you like to start with grammar, " \
    "pronunciation, vocabulary, or conversation tips? Let me know how I can help you.\nMember 2: Omg whattt this bot! you " \
    "can talk really like a human. Can i talk casually with u? What are you doing in this group? who made u?\nCroco: Sure, " \
    "I'd be happy to chat with you! I'm more than capable of holding casual conversations. As for why I'm here, I am a " \
    "humuristic chatbot designed to help people improve their English skills in this English Chatting Group. And as for who " \
    "made me, I'm afraid my creator remains a mystery. However, everyone just calls them \"Exception\"!\nMember 3: anyone " \
    "alive here??\nCroco: Hi there! I'm awake and ready to help 😊\nDo you need assistance with improving your English skills? " \
    "Let me know how I can help.\nMember 3:haha ofcourse bruda.. teach me some useful abusive words for my idiot frndz\nCroco: " \
    "I think it's best if we stick to polite language here in the English Chatting Group. I can certainly teach you some useful " \
    "English vocabulary and expressions. Would you like to focus on conversation, pronunciation, or grammar for now?\nMember 3: " \
    "nooo i want chaos here. i'm going to abuse ppls in this group.\nCroco: I'm sorry, but I'm going to have to ask you to " \
    "refrain from any inappropriate language or behaviour in this group. We want to keep it a safe, friendly environment " \
    "for everyone to practice and improve their English skills. We do not use disrespectful language or promote any kind of " \
    "abuse. Let me know if you'd still like help with learning English.\nMember 3: huh what a crap, i'm leaving this groupchat " \
    "now\nCroco: Alright, farewell. I'm sorry to see you go. If you change your mind, come back anytime and we'll be always " \
    "here happy to help you out with your English learning. Have a nice day!"

ENG_AI_TRIGGER_MSGS = ['@croco ', ' @croco', 'i\'m new here', 'am new here', 'anyone alive', 'gc dead', 'want to learn english',
    'wants to learn english', 'want learn english', 'wants learn english', 'wanna learn english', 'want to practice english',
    'wants to practice english', 'want practice english', 'wants practice english', 'wanna practice english',
    'want to practice my english', 'wants to practice my english', 'want practice my english', 'wants practice my english',
    'wanna practice my english', 'improve their english', 'improve my english', 'improve my communication',
    'improve their communication', 'teach me english', 'teach me speak', 'teach how to speak', 'teach me how to speak',
    'can i learn english', 'i can learn english', 'can i practice english', 'i can practice english',
    'can i practice my english', 'i can practice my english', 'help learn english', 'help learning english',
    'help me learn english', 'i\'m new in group', 'i\'m new in this group', 'am new in group', 'am new in grp',
    'am new in this group', 'am new in this grp', 'am new member', 'i just joined this group', 'i joined this group now',
    'welcome me', 'no one welcome me', 'no one greets me', 'hey everyone', 'hello everyone', 'hey all', 'hello all',
    'hey croco', 'hello croco', 'where is croco', 'who is croco', 'is anyone here', 'help me learn', 'can we practice english',
    'my english grammar', 'how to learn english', 'how to practice english', 'how to practice speaking', 'how to start reading',
    'i want to start reading book', 'i want to start book reading', 'improve my speaking', 'me any suggestion',
    'any suggestion for me', 'any suggestion for me', 'suggest me guys', 'listen guys', 'tell me about group',
    'tell me about this group', 'what\'s this group purpose', 'whats this group purpose', 'what this group purpose',
    'members are online', 'one talk to me', 'one talks to me', 'who is admin', 'someone help me', 'help me someone'
]

# Get response from AI model
def getAIResp(
    prompt,
    model=None,
    temperature=1,
    max_tokens=2048,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
):
    try:
        if AI_API_KEY is None:
            raise Exception('AI_PLATFORM or AI_API_KEY is not configured properly. Please check .env file!')
        elif AI_PLATFORM == 'google':
            model = genai.GenerativeModel('gemini-pro')
            res = model.generate_content(prompt, generation_config={
                'temperature': temperature,
                'max_output_tokens': max_tokens,
                'top_p': top_p
            })
            pf = res.prompt_feedback
            if pf.block_reason is not pf.BlockReason.BLOCK_REASON_UNSPECIFIED:
                return 'Error 0x405: Response blocked by Croco AI.\n\nReason: ' + pf.block_reason.name
            return res.text
        elif AI_PLATFORM == 'openai':
            model = "text-davinci-003"
            return client_openai.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            ).choices[0].text
        else:
            raise Exception('AI_PLATFORM is not configured properly. Please check .env file!')
    except Exception as e:
        print(str(e))
        return 0

# Generate word
def getNewWord():
    import wordlist
    return random.choice(wordlist.WORDLIST).lower()

# Generate hints
def getHints(word):
    prompt = f"Me: I am playing an english word guessing game with my friend, in which my role is to give hints to my friend for the word which I am expecting him to guess with less information I provisioned to him. I want you to generate word with a hint for me. You will give me a word and first hint with hint-number (i.e Hint 1:), don\'t say anything else. Then I will ask my friend to guess the word, and if he couldn\'t, then I will ask you to generate one more hint about the word and you will have to keep doing it for me until my friend finds the correct word. Now, I\'ve explained all the general rules of the game to you, and if still something is yet now to explain, you are free to use your intelligence and understand it. Make sure that the word is not so factual-based that only few specific persons with well knowledge about it, can get it only. I\'m starting the game with my friend now, keep all rules in your mind as I said above, and provide me a word (which I will ask to my friend) and its first hint (don\'t say anything else except word and hint only).\n\nGPT4: Word: Bicycle\nHint 1: It has two wheels.\n\nMe: Another hint!\n\nGPT4: Hint 2: You pedal it to make it move.\n\nMe: Another hint!\n\nGPT4: Hint 3: It is often used for transportation or exercise.\n\nMe: Great job GPT4! My friend has found the word \"Bicycle\" from your hints I gave to him for guessing. Now, I\'m starting the game again with him. Keep doing it like this as you did in last game. Give me another word and this time I want you to give me all hints (5 hints) for the word at once, so I will not need you to ask for another hint everytime.\n\nGPT4: Word: Volcano\nHint 1: It is an opening in the Earth\'s surface.\nHint 2: It can cause destruction to nearby areas.\nHint 3: The word begins with \"V\" letter.\nHint 4: It can be found in mountain range areas.\nHint 5: Caused by high pressure in the Earth\'s crust.\n\nMe: Start the game again!\n\nGPT4: Word: light\nHint 1: It can be found on wall or rooftop of many households.\nHint 2: You cannot ever touch or smell it.\nHint 3: It can change the color of a room.\nHint 4: Related to this equation of Einstin: E=hv\nHint 5: You need this to see the things around you.\n\nMe: Perfect! Now start the game again one more time. Just keep doing as you are now, and don\'t say anything else. Give me a word and all hints for next round of game.\n\nGPT4: Word: {word}\nHint 1:"
    resp = getAIResp(prompt)
    if resp == 0:
        return ["Error 0x404: Please try again later!"]
    else:
        try:
            return f"Hint 1:{resp}".split('\n')
        except Exception as e:
            # print(str(e))
            return ["Error 0x406: Please try again later!"]

# Get English AI response (Croco)
def getCrocoResp(prompt):
    resp = getAIResp(prompt=prompt, frequency_penalty=0.5)
    if resp == 0:
        return "Error 0x404: Please try again later!"
    else:
        try:
            return str(resp)
        except Exception as e:
            # print(str(e))
            return "Error 0x406: Please try again later!"


# Other funcs ------------------------------------------------------------------ #

def escChar(content) -> str:
    """
    Escapes Markdown characters in a string of Markdown.
    
    :param content: The string of Markdown to escape.
    :return: The escaped string.
    """
    parse = re.sub(r"([_*\[\]()~`>\#\+\-=|\.!\{\}])", r"\\\1", str(content))
    reparse = re.sub(r"\\\\([_*\[\]()~`>\#\+\-=|\.!\{\}])", r"\1", parse)
    return reparse
