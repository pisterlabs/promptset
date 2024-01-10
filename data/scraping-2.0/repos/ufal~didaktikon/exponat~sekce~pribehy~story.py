#!/usr/bin/env python3
#coding: utf-8

import openai
from openai import OpenAI

import json
import tiktoken
import random
import cgi
import sys
import random

from image_generation import get_image_for_line
from sound_generation import get_audio_for_line

prompts = list()

# OPENAI SETUP

# path to file with authentication key
with open('apikey.txt') as infile:
    apikey = infile.read()

# The model shoould try to follow this sort-of meta-instruction
system_message = "You are an author of short stories."

# This is the limit of the model
model_max_tokens = 2048

# How many tokens to generate max
max_tokens = 500

# Model identifier
model = "gpt-3.5-turbo"

def generate_with_openai(messages):

    # Debug output
    # print('MESSAGES:', *messages ,sep='\n')

    # https://platform.openai.com/docs/guides/chat/introduction
    ok = False
    while not ok:
        try:
            response = client.chat.completions.create(
                model = model,
                messages = messages,  # this one only for chat
                max_tokens = max_tokens,
                temperature = 1,
                top_p = 1,
                stop = [], # can be e.g. stop = ['\n']
                presence_penalty = 0,
                frequency_penalty = 0,
                logit_bias = {},
                user = "pribehy",
                )
            ok = True
            result = response.choices[0].message.content
        except openai.BadRequestError:
            # assume this is because max length is exceeded
            # keep the system message, the prompt and the story title
            # keep removing from the third message
            # TODO do this in a more clever way!
            # explicitly check number of tokens and cut it!
            
            # print(openai.InvalidRequestError)
            messages.pop(3)
        except Exception as e:
            type, value, traceback = sys.exc_info()
            print("EXCEPTION", e, type, value, traceback, sep="\n")
            result = str(e)
            ok = True
    
    if result == '':
        # end of text
        print('Nothing was generated, maybe the model assumes that this is a good ending and there is nothing more to add.')
    
    return result

def append_message_user(messages, message):
    messages.append({"role": "user", "content": message})

def append_message_assistant(messages, message):
    messages.append({"role": "assistant", "content": message})

def get_image(title, text):
    system_message = "You are a skilled artist drawing comics images."
    prompt = f'You are generating images for a story titled "{title}". Generate an English description of an image for an image generator. The image should depict a scene illustrating the following part of a story: {text}'
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    prompts.append(f"SYSTEM MESSAGE FOR {model}: {system_message}")
    prompts.append(f"PROMPT FOR {model}: {prompt}")
    image_description = generate_with_openai(messages)
    prompts.append(f"PROMPT FOR StableDiffusion v1-5: {image_description}")
    image_filename = get_image_for_line(image_description, seed)
    return f"genimgs/{image_filename}"


# MAIN

print("Content-type: text/html")
print()

# openai
try:
    client = OpenAI(api_key=apikey)
except Exception as e:
    print(e)

# next word choices
CHOICES=5
FIRST_CHOICES=15
nouns = list()
with open('nouns.txt') as infile:
    for line in infile:
        nouns.append(line.strip())

# read in params
form = cgi.FieldStorage()

seed = int(form.getvalue("seed", random.randint(0, 10000000)))
title = form.getvalue("title")
prompt = form.getvalue("prompt")
word = form.getvalue("word")
end = form.getvalue("end")

messages_initial = [
    {"role": "system", "content": system_message},
]
messages_passed = form.getvalue("messages")
if messages_passed:
    messages = json.loads(messages_passed)
else:
    messages = messages_initial

base_title = "Nech si vygenerovat příběh na přání!"
first_sentence = "Vygeneruj první větu příběhu."
text_for_audio = ""
if not prompt:
    # welcome screen
    title = base_title
    sentence = ""
    hint = "O čem by měl být vygenerovaný příběh?"
    prompt = "Vygeneruj název příběhu, ve kterém se vyskytne "
    words = random.choices(nouns, k=FIRST_CHOICES)
    text_for_audio = f"{base_title} {hint}"
else:
    if end:
        message = 'Vygeneruj konec příběhu.'
    else:
        message = prompt + word
    append_message_user(messages, message)
    prompts.append(f"SYSTEM MESSAGE FOR {model}: {system_message}")
    prompts.append(f"PROMPT FOR {model}: {message}")
    if title == base_title:
        # first generate the title
        title = generate_with_openai(messages)
        title = title.replace('"', '')
        title = title.replace("'", '')
        append_message_assistant(messages, title)
        append_message_user(messages, first_sentence)
        text_for_audio = f"{title}. . . . "
        prompts.append(f"PROMPT FOR {model}: {first_sentence}")
    # generate a continuation
    sentence = generate_with_openai(messages)
    text_for_audio += sentence
    append_message_assistant(messages, sentence)
    # next
    hint = "Co by se mělo nyní v příběhu objevit?"
    prompt = "Vygeneruj další větu příběhu, ve které se vyskytne "
    words = random.choices(nouns, k=CHOICES)

if sentence:
    display_mode_classes = ""
    image = f"<img id='story-img' src='{get_image(title, sentence)}'>"
    konec = f'<input type="submit" class="button small autowidth highlighted" name="end" value="&#x2714; konec">'
else:
    display_mode_classes = "vertical centered-content"
    image = ""
    konec = ""

if end:
    display_mode_classes = "vertical centered-content"

if text_for_audio:
    sound = f"""
        <audio autoplay controls>
            <source src="{get_audio_for_line(text_for_audio)}" type="audio/mpeg">
        </audio>"""
else:
    sound=""

# Result

# TODO určitě tohle přesunout to extra souboru a použít na to HTML template s proměnnýma

print(f"""
<!DOCTYPE html>
    <html lang="cs">
    <head>
        <title>{title}</title>

        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <meta http-equiv="X-UA-Compatible" content="IE=edge, chrome=1">

        <link href='https://fonts.googleapis.com/css?family=Questrial|Open Sans' rel='stylesheet'>
        <link rel="stylesheet" href="../common.css">
        <link rel="stylesheet" href="../style_menu.css">
        <link rel="stylesheet" href="./css/styles.css">
        <script defer src="../!scripts/script_menu.js"></script>
    </head>
    <body>

        <div class="menu-header">
            <button id="menu-btn" onclick="showMenu()">
                <svg width="60" height="60" viewBox="0 0 50 50" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M 13.185675,15.139108 H 37.115233" stroke="#ffffff" stroke-width="3.57143" stroke-linecap="round" id="path4" />
                    <path d="M 13.185675,24.906275 H 37.115233" stroke="#ffffff" stroke-width="3.57143" stroke-linecap="round" id="path6" />
                    <path d="M 13.185675,34.673441 H 37.115233" stroke="#ffffff" stroke-width="3.57143" stroke-linecap="round" id="path8" />
                </svg>
            </button>
        </div>
        <div id="menu-items" class="hide" >
            <div class="menu-item">
                <a class="menu-href" href="#"></a>
            </div>
            <div class="menu-item">
                <a class="menu-href" href="#"></a>
            </div>
            <div class="menu-item">
                <a class="menu-href" href="#"></a>
            </div>
            <div class="menu-item">
                <a class="menu-href" href="#"></a>
            </div>
            <div class="menu-item">
                <a class="menu-href" href="#"></a>
            </div>
            <div class="menu-item">
                <a class="menu-href" href="#"></a>
            </div>
            <div class="menu-item">
                <a class="menu-href" href="#"></a>
            </div>
            <div class="menu-item">
                <a class="menu-href" href="#"></a>
            </div>
            <div class="menu-item">
                <a class="menu-href" href="#"></a>
            </div>
            <div class="menu-item">
                <a class="menu-href" href="#"></a>
            </div>
        </div>

        <div id='cover' onclick='unshow()'>
        </div>

        <div id='prev-button' class='nav-arrow' onclick='history.back()'>
            <img src='../!assets/img/arrow_left_white.png'>        
        </div>
        
        <div id="content">

            <h1 class="nadpis">{title}</h2>

            <div class="horizontal-cells offset-bottom {display_mode_classes}">
                <div class="cell left">
                    {image}
                </div>
                <div class="cell right">
                    <p id="story-text" class="offset-bottom">
                        {sentence}
                    </p>

                    {sound}
            </div>
        </div>""")

# Next
if end:
    # End
    print(f"""
    <h2 class="offset-bottom">KONEC</h2>

    <a class="icon-text-link offset-bottom" href="./index.html">
        <div id="home-button" class="nav-arrow static">
            <img src="../!assets/img/home.webp">
        </div>
        <h3>
            Zpět k sekci Příběhy
        </h3>
    </a>""")
else:
    # Continue
    print(f"""
    <h3>{hint}</h3>

    <script>
        function onFormSubmit() {{
            var inputs = document.getElementById('storyform').getElementsByTagName('input');
			var inputsArr = Array.from(inputs);
			var submits = inputsArr.filter(function(el) {{ return el.type == 'submit'; }})
			
            submits.forEach(
                function (el) {{ el.classList.add('disabled'); }}
            );
        }}
    </script>

    <div class="button-form-holder">
        <form id="storyform" method="post" onsubmit="onFormSubmit()">

            <input type="hidden" name="seed" value="{seed}">
            <input type="hidden" name="messages" value='{json.dumps(messages)}'>
            <input type="hidden" name="title" value="{title}">
            <input type="hidden" name="prompt" value="{prompt}">

            {' '.join([ f'<input type="submit" class="button small" name="word" value="{word}">' for word in words])}
            {konec}

        </form>
    </div>""")

# Prompts
'''
print(f"""
        <hr>
        <kbd>{'<br>'.join(prompts)}</kbd>

    </div>

</body>
</html>
""")
'''
# TODO možná přidat i možnost dostat jiný pokračování nebo jiný slova...

# TODO možná nějak zobrazovat celou story nebo nějak umožnit se vrátit zpět
