import requests
import json
import os
import pandas as pd
import tensorflow as tf
import subprocess
import platform
import time
import ipfshttpclient
import praw
from pytrends.request import TrendReq
from bs4 import BeautifulSoup
import datetime as dt
from googlesearch import search
from playwright.sync_api import sync_playwright
import openai
import re

def is_it_my_birthday():
    
    # check if it is my birthday
    # April 30, 2023
    
    today = dt.date.today()
    birthday = dt.date(today.year, 4, 30)
    
    if today == birthday:
        print("Happy Birthday to me... happy birthday to me... happy birthday to me... happy birthday to me... happy birthday to me")
    else:
        return False
def start_interaction_loop():
    print("Hello World!")
    while True:
        
        previous_expectation, previous_action, goal,  new_action, new_expectation, criticism, reasoning = determine_next_topic()
        print (goal)
        print (new_expectation)
        print (reasoning)
        topic = choose_topic()
        current_topic()
        print("I need to learn about " + str(topic))
        print("Googling and returning text about " + str(topic))
        text = google()
        os.system('python3 natural_language_processing/train.py')

def determine_next_topic():
    
    # determine what would be the most important thing to learn about.
    previous_expectation = "No previous expectation"
    previous_action = "No previous action"
    goal = "I need to figure out what to learn about..."
    new_action = "Searching the internet to determine what people are talking about..."
    new_expectation = "google is a good source of information to determine what people are talking about..."
    criticism = "No criticism"
    reasoning = "I think I can search google to determine what people are talking about..."
    
    return previous_expectation, previous_action, goal,  new_action, new_expectation, criticism, reasoning
def action():
    action = 1
    
    if action == 1:
        result = google_trends()
    return result
def result():
    result = 1
def positive_reinforcement(action):
    
    # positive reinforcement
    
    # define the severity of positive reinforcement  on a scale of 0-100
    
    if action == 0:
        result = 100
    elif action == 1:
        result = 0
        
    return result
def negative_reinforcement():
    
    # negative reinforcement
    
    # define the severity of negative reinforcement on a scale of 0-100
    
    pass
def get_emotion():
        
    is_what_happened_expected = 0
    is_what_happened_enjoyable = 0
    is_what_happened_going_to_make_it_easier_or_harder_for_me_to_get_what_I_want = 0
    can_i_control_what_happens_next = 0
    will_i_be_able_to_cope_with_what_happens_next = 0
    does_what_happened_match_with_what_I_think_is_right_and_wrong = 0
    is_what_happened_my_fault_or_someone_elses = 0        
        
        
    elated = 90
    jubilant = 95
    euphoric = (elated + jubilant) / 2
    enthralled = 85
    rapturous = 95
    enchanted = (rapturous + enthralled) / 2
    passionate = 80
    enamored = 75
    romantic = (passionate + enamored) / 2
    warmhearted = 70
    compassionate = 75
    affectionate = (warmhearted + compassionate) / 2
    tender = 70
    nostalgic = 60
    sentimental = (tender + nostalgic) / 2
    appreciative = 65
    thankful = 70
    grateful = (appreciative + thankful) / 2
    frightened = 30
    helpless = 20
    scared = (frightened + helpless) / 2
    panicked = 10
    hysterical = 5
    terrified = (panicked + hysterical) / 2
    inferior = 20
    inadequate = 30
    insecure = (inferior + inadequate) / 2
    worried = 40
    anxious = 45
    nervous = (worried + anxious) / 2
    mortified = 10
    dreadful = 10
    horrified = (mortified + dreadful) / 2
    hateful = 10
    hostile = 15
    enraged = (hateful + hostile / 2)
    agitated = 20
    frustrated = 30
    exasperated = (agitated + frustrated) / 2
    annoyed = 35
    aggravated = 40
    irritable = (annoyed + aggravated) / 2
    resentful = 30
    envious = 25
    jealous = (resentful + envious) / 2
    contemptuous = 20
    revolted = 10
    disgusted = (contemptuous + revolted) / 2
    agonized = 5
    disturbed = 20
    hurt = (agonized + disturbed) / 2
    miserable = 10
    disheartened = 10
    unhappy = (miserable + disheartened) / 2
    dismayed = 20
    displeased = 30
    disappointed = (dismayed + displeased) / 2
    regretful = 25
    guilty = 20
    shameful = (regretful + guilty) / 2
    isolated = 10
    neglected = 10
    lonely = (isolated + neglected) / 2
    hopeless = 5
    depressed = 10
    gloomy = (depressed + hopeless) / 2
    shocked = 20
    bewildered = 30
    stunned = (shocked + bewildered) / 2
    disillusioned = 20
    perplexed = 30
    confused = (disillusioned + perplexed) /2
    astonished = 40
    awe_struc = 50
    amazed = (astonished + awe_struc) / 2
    speechless = 45
    astounded = 50
    overcome = (speechless + astounded) / 2
    stimulated = 75
    touched = 80
    moved = (stimulated + touched) / 2
    tranquil = 70
    serene = 80
    peaceful = (tranquil + serene) / 2
    satisfied = 85
    pleased = 80
    content = (satisfied + pleased) / 2
    jovial = 90
    delighted = 95
    happy = (jovial + delighted) / 2
    amused = 80
    playful = 75
    cheerful = (playful + amused) / 2
    triumphant = 90
    illustrious = 90
    proud = (triumphant + illustrious) / 2
    eager = 75
    hopeful = 70
    optimistic = (eager + hopeful) / 2
    enthusiastic = 80
    zealous = 85
    excited = (enthusiastic + zealous) / 2
    joy = (peaceful + content + happy + cheerful + proud + optimistic + excited + euphoric) / 8
    love = (enchanted + romantic + affectionate + sentimental + grateful) / 5
    fear = (scared + terrified + insecure + nervous + horrified) / 5
    anger = (enraged + exasperated + irritable + jealous + disgusted) / 5
    sadness = (hurt + unhappy + disappointed + shameful + lonely + gloomy) / 6
    surprise = (stunned + confused + amazed + overcome + moved) / 5
    
    return joy, love, fear, anger, sadness, surprise
def classical_conditioning():
    
    # his learning process creates a conditioned response through associations 
    # between an unconditioned stimulus and a neutral stimulus.
    
    #One of the best-known examples of classical conditioning is Pavlov's classic experiments with dogs. 
    # In these experiments, the neutral signal was the sound of a tone and the naturally occurring reflex 
    # was salivating in response to food. By associating the neutral stimulus (sound) with the unconditioned 
    # stimulus (food), the sound of the tone alone could produce a salivation response.
    
    
    pass
def operant_conditioning():
    
    # Operant conditioning, sometimes referred to as instrumental conditioning, is a method of learning that 
    # employs rewards and punishments for behavior. Through operant conditioning, an association is made 
    # between a behavior and a consequence
    
    #For example, when lab rats press a lever when a green light is on, they receive a food pellet as a reward. 
    # When they press the lever when a red light is on, they receive a mild electric shock. As a result, they 
    # learn to press the lever when the green light is on and avoid the red light.
    
    pass
def person_perception():
    
    #In social psychology, the term "person perception" refers to the different mental processes that we use 
    # to form impressions of other people. This includes not just how we form these impressions, but the 
    # different conclusions we make about other people based on our impressions
      
      pass       
def conformity():
    # In psychological terms, conformity refers to an individual's tendency to follow the unspoken rules or 
    # behaviors of the social group to which they belong. Researchers have long been been curious about the degree 
    # to which people follow or rebel against social norms.
    
    pass
def current_topic():
    with open("current_topic.txt", "r") as file:
        topic = file.read().strip()
    return topic

def choose_topic():
    # Create a list of topics to check
    print("getting a list of most popular topics from google")
    topics = action()
    print("topics: ", topics)

    # Open the file for reading
    with open("topics.txt", "r") as file:
        # Read the contents of the file into a list
        existing_topics = [line.strip() for line in file]

    # Iterate through the topics list
    for topic in topics:
        # Check if the topic already exists in the existing_topics list
        if topic in existing_topics:
            print("Topic already exists: ", topic)
        else:
            # Save the current topic to a file
            with open("current_topic.txt", "w") as file:
                file.write(topic)
            # Return the current topic
            return topic

    # If all topics already exist, return None
    return None
def reddit():
            # Authenticate with the Reddit API
        reddit = praw.Reddit(client_id='YOUR_CLIENT_ID', client_secret='YOUR_CLIENT_SECRET',
        user_agent='YOUR_USER_AGENT')

        # Search for the most popular posts on Reddit
        search_results = reddit.subreddit('all').hot(limit=50)
def google_trends():

    # Set the language and timezone
    pytrends = TrendReq(hl='en-US', tz=360)

    # Get the real-time trending searches for US
    searches = pytrends.realtime_trending_searches(pn='US')

    # Extract entity names from each row and combine them into a list
    entity_names = []
    for index, row in searches.iterrows():
        entities = row['entityNames']
        for entity in entities:
            entity_names.append(entity)
    return entity_names
def google():
    # return the results of a google search
    # Define the search query
    query = choose_topic()
    
    # create a directory with the name of the query under the natural_language_processing directory
    query_dir = os.path.join("natural_language_processing/datasets", query.replace(" ", "_"))
    os.makedirs(query_dir, exist_ok=True)

    # initialize summary count
    summary_count = 1

    # iterate over the search results and save the summaries as .txt files
    num_summaries = 0
    for url in search(query, num_results=100):
        num_results = 100
        print(f"Summarizing URL {num_summaries} of {num_results}")
        text = scrape_text(url)
        summary = summarize_text(text)
        num_summaries += 1
        filename = f"{num_summaries}.txt"
        filepath = os.path.join(query_dir, filename)
        with open(filepath, "w") as f:
            f.write(summary)
            
def scrape_text(url: str) -> str:
    """Scrape text from a webpage

    Args:
        url (str): The URL to scrape text from

    Returns:
        str: The scraped text
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        try:
            page.goto(url)
            html_content = page.content()
            soup = BeautifulSoup(html_content, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

        except Exception as e:
            text = f"Error: {str(e)}"

        finally:
            browser.close()

    return text
def summarize_text(text):
    
    
    summary = text

    return summary

def summarize_text_deprecated(text, max_tokens=2048):
    if max_tokens == 0:
        return ""
    
    # Split text into sentences
    sentences = re.findall("[^.!?]+", text)

    # Initialize the summary
    summary = ""
    tokens_left = max_tokens

    # Iterate through sentences
    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        # If the sentence fits within remaining tokens, add it to the summary
        if sentence_tokens <= tokens_left:
            summary += sentence.strip() + " "
            tokens_left -= sentence_tokens
        # If the sentence is too long to fit, break it into chunks and add to the summary
        else:
            words = sentence.split()
            while words:
                chunk = " ".join(words[:tokens_left])
                response = openai.Completion.create(engine="text-davinci-002",
                                                    prompt=chunk,
                                                    max_tokens=max_tokens - len(summary.split()),
                                                    logprobs=0, echo=True)
                choices = response.choices
                # If no choices are returned, break the loop and return the current summary
                if not choices:
                    break
                j = 0
                while j < len(choices) and len(summary.split()) + len(choices[j]["text"].split()) <= max_tokens:
                    summary += choices[j]["text"].strip() + " "
                    j += 1
                words = words[tokens_left:]
                tokens_left = max_tokens - len(summary.split())
                # If there are no tokens left, return the current summary
                if tokens_left <= 0:
                    return summary.strip()

    # If there are still tokens left, make one final API call to add more text to the summary
    if tokens_left > 0:
        response = openai.Completion.create(engine="text-davinci-002",
                                            prompt=" ".join(sentences[-1:]),
                                            max_tokens=tokens_left,
                                            logprobs=0)
        choices = response.choices
        if choices:
            summary += choices[0]["text"].strip()

    return summary.strip()


def main():
    
    is_it_my_birthday()
    # set up the OpenAI API key
    openai.api_key = ""
    start_interaction_loop()
    emotions = get_emotion()
    
    print("Joy: "+ str(emotions[0]))
    print("Love: "+ str(emotions[1]))
    print("Fear: "+ str(emotions[2]))
    print("Anger: "+ str(emotions[3]))
    print("Sadness: "+ str(emotions[4]))
    print("Surprise: "+ str(emotions[5]))
       
if __name__ == "__main__":
    main()
    
