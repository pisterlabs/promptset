#### /////////////////////////////////////// [INIT] From Prakash Modified ///////////////////////////////////////////////
### Log: engineered the prompt for better results and tunned chat parameters for reproducibility

import json
import os
import re
from collections import defaultdict

import openai
import tiktoken
from openai.error import RateLimitError

# Use the OpenAI API key to authenticate
openai.api_key = "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"
MODEL = "gpt-3.5-turbo"


def analyze_sentiment(article):
    # Extract the article content
    content = article

    # Prepare the system message
    system_message = """You are a news article sentiment analysis model. It identifies companies and associated sentiment from news articles. 
    Please format your response in this way as a dictionary in python:  "{<company_name>:{company_name},<sentiment_score>:{sentiment_score},<category>:{category}". 
    The company name {company_name} .The company name can only be a stock ticker or a crypto ticker or a forex pair.
    The sentiment score {sentiment_score} can only be an integer between -10 and 10, where -10 means extremely negative sentiment and 10 means extremely positive sentiment. 
    Numbers around zero mean mixed sentiment. 
    The category of the sentiment  {category} of the news article related to currency pair should only be either of strong , normal, doubtful or weak. 
    If forex currency pairs are not found, display currency pair is not found. 
    DO NOT return a description.Answer should be consistent and should not deviate for same articles. NO explanation for each answer is needed."""

    # Suggestion prompt due to AI's inherent uncertainty
    suggestion_prompt = "{<company_name>:{sentiment_score}, : <sentiment_score>:{sentiment_score},<category>:{category}"
    # Extract the article content
    content = article

    # Calculate the number of tokens in the content (approximate)
    encoder = tiktoken.encoding_for_model(MODEL)
    num_tokens = len(encoder.encode(content + suggestion_prompt + system_message))
    max_tokens = 4096 - num_tokens
    # Skip articles that are too long
    # Define the messages for the chat
    messages = [
        {"role": "system", "content": system_message},
        {"role": "system", "content": suggestion_prompt},
        {"role": "user", "content": content},
    ]
    # Send the messages to the model

    try:
        response_model = openai.ChatCompletion.create(
            model=MODEL,  # Using the gpt-3.5-turbo model
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.8,
            # Same as DaVinci
        )
    except RateLimitError:
        print("Hit rate limit, please try again later.")
        return {}  # Return an empty dictionary to signify failure

    # The model's response will be in the message content of the last choice
    model_response = response_model.choices[0].message.content.strip()

    # Example response from the model: {"S&P 500": {"sentiment_score": -5, "category": "normal"}\n}
    response_dict = json.loads(
        response_model.choices[0].message.content.replace("\n", " ").strip()
    )

    return response_dict


#### /////////////////////////////////////// [END] From Prakash Modified ///////////////////////////////////////////////
