import openai
from config import api_key

openai.api_key = api_key  # Replace with your API key

tweet_text = "@DulwichHistory Loving the complaint about people having to wait 10 minutes for a train.They clearly never travelled via Thameslink."

combined_prompt = f"Analyze this tweet: '{tweet_text}'. First, classify the sentiment in one word as either positive, negative, or neutral. Then, identify the main topic in one word from these options: air conditioning, announcements, brakes, COVID, delays, doors, floor, handrails, hvac, noise, plugs, roof, seats, service, station, tables, tickets/seat reservations, toilets, train general, vandalism, wifi, windows."

def get_combined_analysis_and_token_usage(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI trained to analyze social media content. Provide concise one-word answers for the given tasks."},
            {"role": "user", "content": prompt}
        ]
    )
    analysis = response.choices[0].message['content']
    token_usage = response.usage
    return analysis, token_usage

combined_analysis, token_usage = get_combined_analysis_and_token_usage(combined_prompt)

# Parsing the combined response
sentiment, topic = combined_analysis.split('\n')

print(f"Sentiment: {sentiment}")
print(f"Topic: {topic}")
print(f"Input Tokens: {token_usage['prompt_tokens']}, Output Tokens: {token_usage['completion_tokens']}")
