import openai
import os
import pickle
from alive_progress import alive_bar
from local_settings import secret_key
from SimpleTokenizer import parse_file

openai.api_key = secret_key

def generate_summary(text):
    # Ensure text length is within GPT-3's maximum input size
    if len(text) > 2048:
        text = text[:2048]
    
    # Add specific instructions to the prompt
    prompt = f"The following text is a conversation from a trading transcript: \"{text}\". Summarize the most important trading-related information in four concise bullet points. After the bullet points, provide a brief paragraph suggesting potential actions and key ideas based on the summarized points, all in the style of Warren Buffet. Always penalize redundancy. Write only sentences with full ideas"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.4,
        max_tokens=175
    )

    return response.choices[0].text.strip()


def summarize_stream(Sumstream):
    summarized_parts = [generate_summary(part) for part in Sumstream['parts']]
    return ' '.join(summarized_parts)

filename = 'transcripts.txt'  # or wherever your file is located

# Check if parsed data exists
if os.path.exists('parsed_data.pkl'):
    with open('parsed_data.pkl', 'rb') as f:
        parsed_data = pickle.load(f)
else:
    parsed_data = parse_file(filename)
    # Save parsed data
    with open('parsed_data.pkl', 'wb') as f:
        pickle.dump(parsed_data, f)

# Open a file for writing
with open('summaries.txt', 'a', encoding='UTF-8') as f:
    
    last_processed_title = ""
    if os.path.exists('last_processed.txt'):
        with open('last_processed.txt', 'r', encoding='UTF-8') as lp:
            last_processed_title = lp.read().strip()

    start_processing = False if last_processed_title else True

    with alive_bar(len(parsed_data)) as bar:
        for stream in parsed_data:
            if not start_processing:
                if stream['title'] == last_processed_title:
                    start_processing = True
                continue

            try:
                with open('last_processed.txt', 'w', encoding='UTF-8') as lp:
                    lp.write(stream['title'])
                summary = summarize_stream(stream)
                f.write(f"Title: {stream['title']}\n")
                f.write(f"Summary: {summary}\n\n")
                bar()
            except Exception as e:
                print(f"Error while processing title: {stream['title']}")
                print(str(e))
                break
