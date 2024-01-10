
import openai
from config import api_key

openai.api_key = api_key  # Make sure to use your actual API key

# Tweet text
tweet_text = "@DulwichHistory Loving the complaint about people having to wait 10 minutes for a train.They clearly never travelled via Thameslink."

# Function to call the OpenAI Chat Completions API and print token usage
def get_chat_completion_and_token_usage(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI trained to analyze social media content. Provide answers from the given options."},
            {"role": "user", "content": prompt}
        ]
    )
    message_content = response.choices[0].message['content']
    token_usage = response.usage
    return message_content, token_usage


# Sentiment categories
sentiment_categories = "positive, negative, neutral"

# Topic categories
topic_categories = "air conditioning, announcements, brakes, COVID, delays, doors, floor, handrails, hvac, noise, plugs, roof, seats, service, station, tables, tickets/seat reservations, toilets, train general, vandalism, wifi, windows"

# Getting sentiment with constrained response
sentiment_prompt = f"Classify the sentiment of this tweet as either positive, negative, or neutral: '{tweet_text}'"
sentiment, sentiment_token_usage = get_chat_completion_and_token_usage(sentiment_prompt)

# Getting topic with constrained response
# topic_prompt = f"Identify the main topic of this tweet using only the following options - {topic_categories}: '{tweet_text}'"
topic_prompt = f"Identify the single most likely main topic of this tweet using only the following options: {topic_categories}: '{tweet_text}'"
topic, topic_token_usage = get_chat_completion_and_token_usage(topic_prompt)

print("\n")
print(f"Sentiment Analysis: {sentiment}")
print(f"Topic Extraction: {topic}")
print("\n")
print(f"Sentiment Input Tokens: {sentiment_token_usage['prompt_tokens']}, Sentiment Output Tokens: {sentiment_token_usage['completion_tokens']}")
print(f"Topic Input Tokens: {topic_token_usage['prompt_tokens']}, Topic Output Tokens: {topic_token_usage['completion_tokens']}")




