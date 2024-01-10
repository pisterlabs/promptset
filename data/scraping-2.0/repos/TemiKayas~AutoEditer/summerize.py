import openai
import json

API_KEY=open("API_KEYS.env", "r")
openai.api_key = API_KEY.read()  

def tokenize(text):
    return set(text.lower().split())

def summarize_and_extract_key_quotes_optimized():
    # Load raw_transcript and combine text
    with open("./output/raw_transcript.json", 'r') as file:
        raw_transcript = json.load(file)
    combined_text = ' '.join(raw_transcript["raw_text"])

    # Extract key ideas using GPT-3.5
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize this text: {combined_text}"}
        ]
    )
    key_ideas = response.choices[0].message['content'].split('. ')

    # Print the key ideas
    print("Key Ideas:")
    for idea in key_ideas:
        print("-", idea)

    # Load the full transcript data
    with open("./output/transcript.json", 'r') as file:
        transcript_data = json.load(file)

    matched_ids = []

    for idea in key_ideas:
        idea_tokens = tokenize(idea)

        for entry in transcript_data:
            entry_tokens = tokenize(entry['text'])
            # Check for token overlap
            common_tokens = idea_tokens.intersection(entry_tokens)

            if common_tokens:
                # If there's a significant overlap, consider it a match.
                # The threshold (e.g., 0.5) can be adjusted based on how strict you want the matching to be.
                if len(common_tokens) / len(idea_tokens) > 0.3:
                    matched_ids.append(entry["id"])
                    break  # Break after the first match. If multiple matches are desired, remove this.

    return matched_ids

# Usage
matched_quote_ids = summarize_and_extract_key_quotes_optimized()
print("\nMatched Quote IDs:", matched_quote_ids)
