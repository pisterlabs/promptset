import openai
from config import api_key2

openai.api_key = api_key2  # Replace with your actual API key

tweet_text = "Can't help wishing I'd made Thameslink buy some of those fancy trains that have heating."

combined_prompt = (
    f"Analyze this tweet: '{tweet_text}'. Classify the sentiment as either positive, negative, or neutral. "
    "Identify the main topic from these options: air conditioning, announcements, brakes, COVID, delays, doors, floor, handrails, hvac, "
    "noise, plugs, roof, seats, service, station, tables, tickets/seat reservations, toilets, train general, vandalism, wifi, windows. "
    "Based on the sentiment and topic, suggest an appropriate and specific action."
)

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

def parse_analysis(analysis):
    parts = analysis.split('\n')
    if len(parts) >= 3:
        return parts[0], parts[1], ' '.join(parts[2:])
    else:
        return "Unknown", "Unknown", "No specific action suggested."

combined_analysis, token_usage = get_combined_analysis_and_token_usage(combined_prompt)

# Parsing the combined response
sentiment, topic, suggested_action = parse_analysis(combined_analysis)

print(f"{sentiment}")
print(f"{topic}")
print(f"Suggested Action: {suggested_action}")
print(f"Input Tokens: {token_usage['prompt_tokens']}, Output Tokens: {token_usage['completion_tokens']}")