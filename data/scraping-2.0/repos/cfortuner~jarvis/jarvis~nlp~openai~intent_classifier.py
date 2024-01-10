import os
import random
from typing import Any

import openai

from jarvis.nlp import nlp_utils

from . import caching

openai.api_key = os.getenv("OPENAI_API_KEY")

CATEGORIES = ["Messaging", "Website", "Other"]
messaging = [
    # Messaging
    ("tell Mom I'm coming home for dinner", "Messaging"),
    ("Let Hari know I just pushed my latest changes to the github repo", "Messaging"),
    ("msg Jackie and let her know I'll be home by 10 tonight", "Messaging"),
    ("Reply Sounds fun!", "Messaging"),
    ("text Colin on Facebook Messenger and ask him if he's free for tennis tomorrow", "Messaging"),
    ("Want to hang out tonight?", "Messaging"),

]
web_nav = [
    ("Go to my amazon cart", "Website"),
    ("open my github pull requests", "Website"),
    ("search wikipedia for grizzly bears", "Website"),
    ("search amazon for ski mask filter for Prime only", "Website"),
    ("Sign out of my account", "Website"),
    ("Login to my new york times account", "Website"),
    ("search for hard-shell rain jackets on ebay", "Website"),
    ("open wikipedia", "Website"),
    ("log out", "Website"),
    ("leetcode.com", "Website"),
]
other = [
    ("how are you doing?", "Other"),
    ("that's interesting. are you going to come over later?", "Other"),
    ("switch to chrome", "Other"),
    ("merge my pull request", "Other"),
    ("call Dan and ask him if he's coming to Desolation", "Other"),
    ("order takeout from La Tacos to 102 clipper st.", "Other"),
    ("ok that's about right", "Other"),
    ("Hello Higgins", "Other")
]
EXAMPLES = messaging + web_nav + other
random.Random(4).shuffle(EXAMPLES)


def classify_intent_completion(text: str, engine: str = "davinci", cache: Any = None) -> str:
    prompt = """Classify the following commands into categories:\n"""
    for command, category in EXAMPLES:
        prompt += f"\nCommand: {command}\nCategory: {category}"
    prompt += f"\nCommand: {text}\nCategory:"

    cache = cache if cache is not None else caching.get_default_cache()
    cache_key = nlp_utils.hash_normalized_text(prompt)
    if cache_key not in cache:
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=0,
            max_tokens=2,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"]
        )
        answer = response["choices"][0]["text"].strip()
        cache.add(
            key=cache_key,
            value={
                "cmd": text,
                "answer": answer,
                "response": response
            }
        )
    else:
        answer = cache[cache_key]["answer"]
        response = cache[cache_key]["response"]

    if answer not in CATEGORIES:
        print("Answer: {answer} not in supported intent categories.")
        return "Other"

    return answer


def classify_intent_classification(text: str, cache: Any = None) -> str:
    cache = cache if cache is not None else caching.get_default_cache()
    cache_key = nlp_utils.hash_normalized_text(text)
    if cache_key not in cache:
        response = openai.Classification.create(
            search_model="ada",
            model="curie",
            examples=EXAMPLES,
            query=text,
            labels=CATEGORIES,
        )
        answer = response["label"].strip()
        cache.add(
            key=cache_key,
            value={
                "cmd": text,
                "answer": answer,
                "response": response
            }
        )
    else:
        answer = cache[cache_key]["answer"]
        response = cache[cache_key]["response"]

    return answer


if __name__ == "__main__":
    examples = [
        # WebNav
        ("sign in to my yahoo account", "Website"),
        ("go to target.com", "Website"),
        ("find me airpods on ebay", "Website"),
        ("search wikipedia", "Website"),
        ("search google", "Website"),
        ("search bing for Harley-Davidson motorcycles", "Website"),
        # Messaging
        ("let Brian Fortier know I'm coming to graduation", "Messaging"),
        ("text Brendan on WhatsApp", "Messaging"),
        ("send Colin an email saying things are good", "Messaging"),
        # Other
        ("wondering what the air quality will be", "Other"),
        # It gets this wrong. We need more context about whether the user means the Spotify website or the Spotify App
        # ("play Stan Getz on Spotify", "Other"),
        ("turn on the lights", "Other"),
        ("close all applications", "Other"),
        ("which app is using the most CPU?", "Other"),
    ]
    print("classify_intent_completion -------------")
    for example, category in examples:
        predicted = classify_intent_completion(example)
        if predicted != category:
            print(f"Cmd: {example}, Predicted: {predicted}, Expected: {category}")

    print("classify_intent_classification -------------")
    for example, category in examples:
        predicted = classify_intent_classification(example)
        if predicted != category:
            print(f"Cmd: {example}, Predicted: {predicted}, Expected: {category}")
