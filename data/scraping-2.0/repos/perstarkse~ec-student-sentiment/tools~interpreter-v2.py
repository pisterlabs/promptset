import argparse
import json
import os
import re
from bs4 import BeautifulSoup
from collections import defaultdict
from dotenv import load_dotenv
import openai
from datetime import datetime

def split_text(text, size_limit):
    words = text.split()
    chunks = []
    chunk = []
    size = 0

    for word in words:
        word_size = len(word)
        if size + word_size > size_limit:
            chunks.append(" ".join(chunk))
            chunk = [word]
            size = word_size
        else:
            chunk.append(word)
            size += word_size

    chunks.append(" ".join(chunk))

    return chunks

def parse_chatlog(html):
    soup = BeautifulSoup(html, 'html.parser')
    messages_by_week = defaultdict(list)
    total_messages = 0
    counted_messages = 0

    for message_group in soup.find_all('div', {'class': 'chatlog__message-group'}):
        message_data = {}
        message_container = message_group.find('div', {'class': 'chatlog__message-container'})
        total_messages += 1

        # Try finding timestamp in the chatlog__message div first
        timestamp_element = message_container.find('div', {'class': 'chatlog__message'}).find('span', {'class': 'chatlog__timestamp'})
        if timestamp_element is None:
            # If not found, try finding it in the chatlog__message-container div
            timestamp_element = message_container.find('span', {'class': 'chatlog__timestamp'})
        
        if timestamp_element is not None:
            timestamp = timestamp_element.text.strip()
            message_data['timestamp'] = timestamp
            date_str = timestamp.split()[0]
            date_obj = datetime.strptime(date_str, '%m/%d/%Y')
            week_number = date_obj.isocalendar()[1]
            counted_messages += 1
        else:
            continue  # Skip this message if there's no timestamp

        content = message_container.find('div', {'class': 'chatlog__content chatlog__markdown'})
        if content:
            message_data['message'] = content.text.strip()
        
        messages_by_week[week_number].append(message_data)
    print(f"Total messages: {total_messages}")
    print(f"Counted messages: {counted_messages}")
    return messages_by_week

def analyze_text(text, api_key, prompt, max_retries=1):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        temperature=0,
    )
    result = response['choices'][0]['message']['content'].strip().replace('\n', '')
    match = re.search(r'\{.*\}', result)
    if match:
        json_result = match.group()
        return json_result
    else:
        print(f"No JSON found in the output: {result}")
        if max_retries > 0:
            print(f"Retrying... Remaining retries: {max_retries - 1}")
            return analyze_text(text, api_key, prompt, max_retries - 1)
        else:
            print(f"No JSON found in the output: {result}")
            default_scores = {
                'Demand': 50,
                'Control': 50,
                'Support': 50,
                'Sentiment': 50,
                'Engagement': 50
            }
            return json.dumps(default_scores)
        
def consolidate_data(results_list):
    consolidated_data = {}
    for data in results_list:
        week = data['week']
        if week not in consolidated_data:
            consolidated_data[week] = data
            consolidated_data[week]['count'] = 1
        else:
            consolidated_data[week]['num_messages'] += data['num_messages']
            consolidated_data[week]['demand'] += data['demand']
            consolidated_data[week]['control'] += data['control']
            consolidated_data[week]['support'] += data['support']
            consolidated_data[week]['sentiment'] += data['sentiment']
            consolidated_data[week]['engagement'] += data['engagement']
            consolidated_data[week]['count'] += 1

    for week, data in consolidated_data.items():
        data['demand'] = round(data['demand'] / data['count'])
        data['control'] = round(data['control'] / data['count'])
        data['support'] = round(data['support'] / data['count'])
        data['sentiment'] = round(data['sentiment'] / data['count'])
        data['engagement'] = round(data['engagement'] / data['count'])
        del data['count']

    return list(consolidated_data.values())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse, analyze and process chatlog data.')
    parser.add_argument('html_file', help='Path to the HTML file containing chatlog data.')
    args = parser.parse_args()

    with open(args.html_file, 'r', encoding='utf-8') as f:
        html = f.read()
    parsed_messages = parse_chatlog(html)

    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    prompt = """
    Analyze the text and assign scores (0-100) for 'Demand', 'Control', 'Support', 'Sentiment', and 'Engagement' based on the Demand Control Support Model and sentiment analysis. Do your best attempt at interpreting, no excuses! Output the scores in JSON format. No other output is allowed.
    """
    results_list = []

    for week_number, messages in parsed_messages.items():
        text = ' '.join([message['message'] for message in messages if 'message' in message])
        chunks = split_text(text, 3500)
        for chunk in chunks:
            result = analyze_text(chunk, api_key, prompt)
            if result is not None:
                result_json = json.loads(result.encode('utf-8'))
                analysis_result = {
                    'week': week_number,
                    'num_messages': len(messages),
                    'demand': result_json.get('Demand', None),
                    'control': result_json.get('Control', None),
                    'support': result_json.get('Support', None),
                    'sentiment': result_json.get('Sentiment', None),
                    'engagement': result_json.get('Engagement', None),
                }
                results_list.append(analysis_result)

    results_list = consolidate_data(results_list)

    output_filename = os.path.splitext(os.path.basename(args.html_file))[0] + '_analysis.json'
    os.makedirs('output', exist_ok=True)
    with open(os.path.join('output', output_filename), 'w', encoding='utf-8') as f:  
        json.dump(results_list, f, ensure_ascii=False, indent=4)
    print(f'Successfully saved the analysis results to output/{output_filename}!')


