#%%#
from cgitb import text
import discord
from aiogoogle import Aiogoogle
import openai
import json
import textwrap
import numpy as np
import os
from IPython.display import display, HTML
import aiohttp
import snscrape.modules.twitter as sntwitter

class DictToObject(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

os.chdir(os.path.expanduser("~/Documents/Projects/Sunflower/SunflowerOS/Repo/VibesEnforcer"))

secrets = json.load(open("secrets.json", "rb"))
openai.api_key = secrets["openaiApiKey"]

async def getCompletionOAI(*args, **kwargs):
    # kwargs = {"prompt": "test with space at end "}
    # kwargs = {"prompt": "testing123"}
    if kwargs["prompt"][-1] == " ":
        raise ValueError("Trailing space in prompt")

    defaultArgs = dict(
        #engine="text-davinci-002",
        model="text-davinci-002",
        temperature=0.7,
        max_tokens=120,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["Message from"],
    )
    combinedArgs = {**defaultArgs, **kwargs}
    url = "https://api.openai.com/v1/completions"
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {secrets['openaiApiKey']}",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=combinedArgs, headers=headers) as r:
            responseBody = await r.json()
    responseText = responseBody["choices"][0]["text"].strip()
            
    # response = openai.Completion.create(**combinedArgs)
    # responseText = response.choices[0].text.strip()

    promptDebug = ""
    promptDebug += str(kwargs["prompt"])
    if "suffix" in kwargs:
        promptDebug += "[INSERT]" + str(kwargs["suffix"])
    promptDebug+= "\n" + "--ARGS:" + str({i:combinedArgs[i] for i in combinedArgs if i!='prompt'})
    fullArgs = str(combinedArgs)

    prediction = {"promptDebug": promptDebug, "responseText": responseText, "fullArgs": fullArgs}

    return prediction

scraper = sntwitter.TwitterTweetScraper(tweetId=1553200773164462080)
out = list(scraper.get_items())

async def reactToSignalAgreeUrge(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message):
    prompt = textwrap.dedent(f"""
    In their second message, {a1Name} adds onto their first message, conveying to the reader that people who agree with their first message are peers who they respect and would work with, and strongly implying that they don't think they're better than anyone else.
    \t
    {a1Name}'s Profile:
    Follower Count: {a1FollowerCt}, Following Count: {a1FollowingCt}
    Bio: {a1Bio}
    \t
    Message from {a1Name}: 
    {a1Message}
    \t
    Message from {a1Name}:
    {a1Message}
    """).strip() + "\n"
    response = await getCompletionOAI(prompt=prompt, temperature=0.7)
    return response

async def reactToBragging(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message):
    prompt = textwrap.dedent(f"""
    In their second message, {a1Name} is much more humble, and is vulnerable in explaining why they sent the first message.
    \t
    {a1Name}'s Profile:
    Follower Count: {a1FollowerCt}, Following Count: {a1FollowingCt}
    Bio: {a1Bio}
    \t
    Message from {a1Name}: 
    {a1Message}
    \t
    Message from {a1Name}: 
    """).strip() 
    response = await getCompletionOAI(prompt=prompt, temperature=0.7)
    return response

async def reactToDoom(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message):

    prompt = textwrap.dedent(f"""
    The following is a message from {a1Name}. His first message is pessimistic, but his second message adds onto the first and is very optimistic and inspiring.
    \t
    {a1Name}'s Profile:
    Follower Count: {a1FollowerCt}, Following Count: {a1FollowingCt}
    Bio: {a1Bio}
    \t
    Message from {a1Name}: 
    {a1Message}
    \t
    Message from {a1Name}:
    {a1Message}
    """).strip() + "\n"
    response = await getCompletionOAI(prompt=prompt, temperature=0.7)
    return response

async def reactToExcitementFOMO(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message):

    prompt = textwrap.dedent(f"""
    The following is a message from {a1Name}. In his second message, {a1Name} tones down the hype in their first message, encouraging the reader to stay focused on what they are working on.
    \t
    {a1Name}'s Profile:
    Follower Count: {a1FollowerCt}, Following Count: {a1FollowingCt}
    Bio: {a1Bio}
    \t
    Message from {a1Name}: 
    {a1Message}
    \t
    Message from {a1Name}:
    """).strip() 
    response = await getCompletionOAI(prompt=prompt, temperature=0.7)
    print(response["promptDebug"])
    print("\n" + response["responseText"])

    return response

async def reactToPolarizeAgree(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message, a2Name, a2Bio, a2FollowerCt, a2FollowingCt, a2Message):

    prompt1 = textwrap.dedent(f"""
    In this conversation, {a2Name} apologizes, is vulnerable and explains to {a1Name} how they really feel and why.
    \t
    Message from {a2Name}:
    {a2Message}
    \t
    Message from {a1Name}:
    {a1Message}
    \t
    Message from {a2Name} detailing their feelings:
    """).strip() 
    response1 = await getCompletionOAI(prompt=prompt1, temperature=0.2)
    a2Feeling = response1["responseText"]

    prompt2 = textwrap.dedent(f"""
    In this conversation, {a1Name} apologizes, is vulnerable, and explains to {a2Name} how they really feel and why.
    \t
    Message from {a2Name}:
    {a2Message}
    \t
    Message from {a1Name}:
    {a1Message}
    \t
    Message from {a2Name}:
    {a2Feeling.replace(chr(10), "")}
    \t
    Reply from {a1Name}, detailing their feelings:
    """).strip()
    # print(prompt2)
    response2 = await getCompletionOAI(prompt=prompt2, temperature=0.2)
    a1Feeling = response2["responseText"]
    # print(a1Feeling)

    return [response1, response2]


# "why they wrote what they wrote" works well when already agree with a2name (i.e. disagree with a1Name)
# When there's a dunk, the person who dunked is the first person, and the person who got dunked on is the second person, and the reader agrees with the target and feels attacked.
async def reactToPolarizeDisagree(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message, a2Name, a2Bio, a2FollowerCt, a2FollowingCt, a2Message):

    prompt1 = textwrap.dedent(f"""
    In this conversation, {a2Name} apologizes, is vulnerable and explains to {a1Name} why they wrote what they wrote.
    \t
    Message from {a2Name}:
    {a2Message}
    \t
    Message from {a1Name}:
    {a1Message}
    \t
    Message from {a2Name} detailing their feelings and explaining why they wrote what they wrote:
    """).strip() 
    # print(prompt)
    response1 = await getCompletionOAI(prompt=prompt1, temperature=0.2)
    a2Feeling = response1["responseText"]
    # print(response["responseText"])

    prompt2 = textwrap.dedent(f"""
    In this conversation, {a1Name} apologizes, is vulnerable, and explains to {a2Name} how they really feel and why.
    \t
    Message from {a2Name}:
    {a2Message}
    \t
    Message from {a1Name}:
    {a1Message}
    \t
    Message from {a2Name}:
    {a2Feeling.replace(chr(10), "")}
    \t
    Reply from {a1Name}, detailing their feelings:
    """).strip()
    # print(prompt2)
    response2 = await getCompletionOAI(prompt=prompt2, temperature=0.2)
    a1Feeling = response2["responseText"]
    # print(a2Feeling)

    return [response1, response2]

async def createThoughtguideFromTweet(url):
    # url = "https://twitter.com/dystopiabreaker/status/1553200773164462080" #superior
    # url = "https://twitter.com/fchollet/status/976933782367293440" #doom
    # url = "https://twitter.com/hasanthehun/status/1562554743846604800" #argument
    # url = "https://twitter.com/Alariko_/status/1562448991086051332" #bragging
    # url = "https://twitter.com/makeitrad1/status/1562815837475467264" #hype/fomo
    # url = "https://twitter.com/theartofasty/status/1563603627809222657" #negativity-ethics


    tweetId = int(url.split("/")[-1])
    scraper = sntwitter.TwitterTweetScraper(tweetId=tweetId)
    tweetInfo = list(scraper.get_items())[0]
    a1Name = tweetInfo.user.username
    a1Bio = tweetInfo.user.description
    a1FollowerCt = tweetInfo.user.followersCount
    a1FollowingCt = tweetInfo.user.friendsCount
    a1Message = tweetInfo.content.replace("\n", " ") # temp fix for newlines screwing up f string indentation
    if tweetInfo.card is not None:
        a1Message += f""" [CARD: {tweetInfo.card.siteUser.displayname} | {tweetInfo.card.title} | {tweetInfo.card.description}]"""

    if tweetInfo.quotedTweet is not None:
        a2Name = tweetInfo.quotedTweet.user.username
        a2Bio = tweetInfo.quotedTweet.user.description
        a2FollowerCt = tweetInfo.quotedTweet.user.followersCount
        a2FollowingCt = tweetInfo.quotedTweet.user.friendsCount
        a2Message = tweetInfo.quotedTweet.content.replace("\n", " ")
        if tweetInfo.quotedTweet.card is not None:
            a2Message += f""" [CARD: {tweetInfo.quotedTweet.card.siteUser.displayname} | {tweetInfo.quotedTweet.card.title} | {tweetInfo.quotedTweet.card.description}]"""
    
    classificationOutput = await classifyWithPrompt(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message)
    print(f"""{a1Name}: {a1Message}\n\n{classificationOutput["responseText"]}""")
    # print(classificationOutput["promptDebug"])

    classString = classificationOutput["responseText"].lower()

    # TODO: deal with cases that depend on reader's beliefs
    classString = classString.split("\n")[0]
    classification = ""


    if len(classString.split("\n")) == 1:
        if "doom" in classString:
            print("*Using doom action path*\n")
            classification = "doom"
            thoughtGuideOutput = await reactToDoom(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message)
            thoughtGuide = thoughtGuideOutput["responseText"]
        elif "superior" in classString or "must-share" in classString:
            print("*Using superior / shareurge action path*\n")
            classification = "superior"
            thoughtGuideOutput = await reactToSignalAgreeUrge(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message)
            thoughtGuide = thoughtGuideOutput["responseText"]
        elif "bragging" in classString:
            print("*Using bragging action path*\n")
            classification = "bragging"
            thoughtGuideOutput = await reactToBragging(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message)
            thoughtGuide = thoughtGuideOutput["responseText"]
        elif "excitement-fomo" in classString:
            print("*Using excitement-fomo action path*\n")
            classification = "excitement-fomo"
            thoughtGuideOutput = await reactToExcitementFOMO(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message)
            thoughtGuide = thoughtGuideOutput["responseText"]
        elif "agree-dunk" in classString:
            print("*Using agree-dunk action path*\n")
            classification = "agree-dunk"
            # Goal is to make the reader have empathy with the person being dunked on

            if "a2Name" not in locals():
                a2Name = "B"
                a2Message = "hi"
                a2FollowerCt = 100
                a2FollowingCt = 100
                a2Bio = "hello everyone! this is my bio"
            # TODO: won't necessarily be a2name & quoted tweet, e.g. https://twitter.com/ReinH/status/1563210768446734336
            #also, quote-tweet isn't necessarily the one being dunked on, e.g. https://twitter.com/Post__Curtis/status/1562559162248097792

            thoughtGuideOutput = await reactToPolarizeAgree(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message, a2Name, a2Bio, a2FollowerCt, a2FollowingCt, a2Message)
            thoughtGuide = f"Response from {a2Name}: " + thoughtGuideOutput[0]["responseText"] + f"\n\nResponse from {a1Name}: " +thoughtGuideOutput[1]["responseText"] 
        elif "attack-anger" in classString:
            print("*Using attack-anger action path*\n")
            classification = "attack-anger"
            # Goal is to make the reader have empathy with the person doing the attacking

            if "a2Name" not in locals():
                a2Name = "RDR"
                a2Message = "hi"
                a2FollowerCt = 100
                a2FollowingCt = 100
                a2Bio = "hello everyone! this is my bio"

            thoughtGuideOutput = await reactToPolarizeDisagree(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message, a2Name, a2Bio, a2FollowerCt, a2FollowingCt, a2Message)
            thoughtGuide = f"Response from {a2Name}: " + thoughtGuideOutput[0]["responseText"] + f"\n\nResponse from {a1Name}: " +thoughtGuideOutput[1]["responseText"] 
        else:
            classification = "other <" + classString + ">"
            thoughtGuide = "None"
        print(thoughtGuide)
    return {"thoughtGuide": thoughtGuide, "classification": classification, "promptDebug1": classificationOutput["promptDebug"]}

async def classifyWithPrompt(a1Name, a1Bio, a1FollowerCt, a1FollowingCt, a1Message):
    prompt = textwrap.dedent(f"""
        Describe which of the following categories the message from {a1Name} falls into for PersonC.
        \t
        -- Categories --
        Doom: the message creates a senes of doom or pessimism about the current state of affairs.
        Excitement-FOMO: the message makes PersonC excited about a new technology or opportunity, but also makes them feel that they should be doing something and that they are missing out.
        Must-Share-Too: the message makes PersonC feel that they need to tell everyone that they agree, like a student in class feeling the need to share that they know something right before the teacher teaches everyone. It makes the reader feel the need to let everyone know that they had the thought first.
        Unfair: the message makes PersonC feel that they are being wronged, that something unfair is happening.
        Agree-Dunk: the message is dunking on someone, and PersonC feels a sense of accomplishment that their team is winning.
        Anger-Dunk: the message is dunking on someone, and PersonC feels attacked because they side with the person being dunked on.
        Attack-Anger: the message makes PersonC feel that some part of their identity is being attacked, even if it is just implied in the message.
        Bragging: the message comes across as bragging or a humble-brag, and might make the reader feel less skilled.
        Superiority-Agreement: the message makes the PersonC feel like the author is trying to be superior to them specifically because the author is sharing an idea first, a sort of nerdsniping.
        Insecure-Fear: the message makes PersonC worry about some aspect about themselves which they are insecure about, like their appearance.
        Project-Vulernability: the message is from PersonC, and they are sharing a project that they are working on, and they are being vulnerable because the project is not finished.
        Fortune-Cookie: the message is satisfying to read and feels like it is offering wisdom, but doesn't really say anything concrete.
        \t
        -- Background Information --
        {a1Name}'s Profile:
        Follower Count: {a1FollowerCt}, Following Count: {a1FollowingCt}
        Bio: {a1Bio}
        \t
        -- Messages --
        Message from {a1Name}: 
        {a1Message.replace(chr(10), " ")}
        \t
        -- Classification --
        Which category does the message from {a1Name} fall into for PersonC? If it depends on PersonC's beliefs, list the possible scenarios in the form [if belief]: [category].
        \t
        1.
    """).strip()
    response = await getCompletionOAI(prompt=prompt, temperature=0.0, max_tokens=300)
    return response
# %%

# If site defined, stop it, so can easily refresh server
if 'site' in globals():
    await site.stop()

# Async aiohttp web app
from aiohttp import web
import logging
logging.basicConfig(level=logging.INFO)
import aiohttp_cors
import asyncio
app = web.Application()
routes = web.RouteTableDef()

@routes.get('/')
async def index(request):
    return web.Response(text="Hello, world")
@routes.post('/generate')
async def generate(request):
    data = await request.json()
    response = await createThoughtguideFromTweet(data["tweetURL"])
    return web.json_response(response)
app.add_routes(routes)

# Send CORS header on all requests
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*"
    )
})
for route in list(app.router.routes()):
    cors.add(route)

runner = aiohttp.web.AppRunner(app)
await runner.setup()
site = aiohttp.web.TCPSite(runner, 'localhost', 8089)    
await site.start()

# %%
