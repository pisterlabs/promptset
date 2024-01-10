#%%
# Cohere Demo: Classify Endpoint basic example (no custom model) 
# Source: https://docs.cohere.com/docs/classify-endpoint

import os
from dotenv import load_dotenv

import cohere
from cohere.responses.classify import Example


#%% 
# Connect

load_dotenv()
api_key = os.getenv('cohere_key__free_trial') 
co = cohere.Client(api_key)


#%% 
# Sample Training & Scoring Data
examples = [Example("I'm so proud of you", "positive"), 
            Example("What a great time to be alive", "positive"), 
            Example("That's awesome work", "positive"), 
            Example("The service was amazing", "positive"), 
            Example("I love my family", "positive"), 
            Example("They don't care about me", "negative"), 
            Example("I hate this place", "negative"), 
            Example("The most ridiculous thing I've ever heard", "negative"), 
            Example("I am really frustrated", "negative"), 
            Example("This is so unfair", "negative"),
            Example("This made me think", "neutral"), 
            Example("The good old days", "neutral"), 
            Example("What's the difference", "neutral"), 
            Example("You can't ignore this", "neutral"), 
            Example("That's how I see it", "neutral")            
            ]

inputs=["Hello, world! What a beautiful day",
        "It was a great time with great people",
        "Great place to work",
        "That was a wonderful evening",
        "Maybe this is why",
        "Let's start again",
        "That's how I see it",
        "These are all facts",
        "This is the worst thing",
        "I cannot stand this any longer",
        "This is really annoying",
        "I am just plain fed up"
        ]


#%% 
# Classify on the fly

def classify_text(inputs, examples):

    response = co.classify(
        model='embed-english-v2.0',
        inputs=inputs,
        examples=examples)
  
    classifications = response.classifications

    return classifications

predictions = classify_text(inputs,examples)

predictions


#%%

for index, text in enumerate(inputs):
    print(f"Item {index+1}: {text} --> {predictions[index].predictions} (Confidence={predictions[index].confidences})")



#%%
