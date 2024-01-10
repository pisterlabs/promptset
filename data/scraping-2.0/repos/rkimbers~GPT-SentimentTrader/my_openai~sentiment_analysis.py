import os
import openai
import re
import logging

from openai.error import OpenAIError, RateLimitError
from collections import defaultdict


def analyze_sentiment(article):
    # Use the OpenAI API key to authenticate
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Extract the article content
    content = article['content']

    # Calculate the number of tokens in the content (approximate)
    num_tokens = len(content) // 4  # one token ~= 4 characters

    # Skip articles that are too long
    if num_tokens > 4096:  # current model (gpt-3.5-turbo) has a limit of 4096 tokens
        logging.info(f"Skipping article with {num_tokens} tokens.")
        return {}

    # Prepare the system message
    system_message = """This is a news article sentiment analysis model. It identifies companies and associated sentiment from news articles. 
    Please format your response in this way: (replace with company's name): (sentiment score integer). 
    The sentiment score can only be an integer between -10 and 10, where -10 means extremely negative sentiment and 10 means extremely positive sentiment. The number MUST be between -10 and 10. 
    Numbers around zero mean mixed sentiment. DO NOT return a description. If no company is found, please state that no company is found."""

    # Suggestion prompt due to AI's inherent uncertainty
    suggestion_prompt = "Company: Integer"

    # Prepare the user message (the article content)
    user_message = content

    # Define the messages for the chat
    messages = [
        {"role": "system", "content": system_message},
        {"role": "system", "content": suggestion_prompt},
        {"role": "user", "content": user_message},
    ]
    
    # Send the chat messages to the GPT API and get the response
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using the gpt-3.5-turbo model
            messages=messages,
            max_tokens=1000,
            temperature=0.2# Same as DaVinci
        )
    except RateLimitError:
        logging.error("Hit rate limit, please try again later.")
        return {}  # Return an empty dictionary to signify failure
    except OpenAIError as e:
        logging.error(f"OpenAI error: {e}")
        return {}

    # The model's response will be in the message content of the last choice
    model_response = response.choices[0].message.content.strip()

    logging.info(model_response)

    # Process the response
    scores = defaultdict(list)
    try:
        responses = re.split(",|\\.", model_response)  # Separate different company-score pairs by comma or period
        for response in responses:
            response = response.strip()  # Remove leading/trailing whitespace
            if ":" in response:
                parts = response.split(":")
                company = parts[0].strip()
                sentiment_score = int(re.findall(r"-?\d+", parts[1].strip())[0])  # handle numbers with a trailing period
                scores[company].append(sentiment_score)
                
    except ValueError:
        logging.error(f"Could not process the model's response: {model_response}")
        
    return scores

