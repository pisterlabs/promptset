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
    #prompt = f"{text}\n\nSummarize the following text using bullet points to keep only the most important information for trading related knowledge. Be as concise as possible so you can achieve to summarize as much relevant data and highlights from the transcript:"
        # Add specific instructions to the prompt
    #prompt = f"{text}\n\nAI, summarize the key trading-related insights from the above text. Please provide a concise summary in bullet points highlighting the most important details:"
        # Add specific instructions to the prompt
    #prompt = f"{text}\n\Summarize the key insights from the above text. Please provide a relevant summary in bullet points highlighting the most important details. After finishing, provide a brief and conclusion"
    # Add specific instructions to the prompt
    #prompt = f"{text}\n\nThe following is a transcript relevant to trading. Summarize this information concisely and extract the most important points. Aim for a style similar to Warren Buffet's and present the summary as bullet points:"
    # Add specific instructions to the prompt
    #prompt = f"The following text is a conversation from a trading transcript: \"{text}\". Summarize the most important trading-related information in three concise bullet points. After the bullet points, provide a brief conclusion suggesting potential actions and key ideas based on the summarized points, all in the style of Warren Buffet."
    # Add specific instructions to the prompt
    #prompt = f"The following text is a conversation from a trading transcript: \"{text}\". Summarize the most important trading-related information in five concise bullet points. After the bullet points, provide a brief paragraph suggesting potential actions and key ideas based on the summarized points, all in the style of Warren Buffet. Always penalize redundancy. Write only sentences with full ideas"
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

# Define N
N = 2

# Open a file for writing
with open('summaries.txt', 'a', encoding='UTF-8') as f:
    
    last_processed_title = ""
    if os.path.exists('last_processed.txt'):
        with open('last_processed.txt', 'r', encoding='UTF-8') as lp:
            last_processed_title = lp.read().strip()

    start_processing = False if last_processed_title else True
    counter = 0

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
                counter += 1
                if counter >= N:
                    break
            except Exception as e:
                print(f"Error while processing title: {stream['title']}")
                print(str(e))
                break
