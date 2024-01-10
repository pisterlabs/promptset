import json
import requests
from bs4 import BeautifulSoup
import argparse
import time

def parse_args():
    '''Parses the command line arguments'''
    parser = argparse.ArgumentParser(description='Scrape a conversation from OpenAI chat')
    parser.add_argument('--input_url', type=str, help='The URL of the conversation to scrape', default=None)
    parser.add_argument('--input_file', type=str, help='The file containing the URLs of the conversations to scrape', default=None)
    return parser.parse_args()

def fetch_webpage_content(input_url):
    '''Fetches the content of a webpage'''
    # Headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    # Creating a session to handle cookies
    with requests.Session() as session:
        # Send a get request to the URL
        response = session.get(input_url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Use BeautifulSoup to parse the HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
        else:
            return f"Error fetching the page: Status code {response.status_code}"


def get_conversation(soup):
    '''Extracts the conversation from the webpage'''
    script_tag = soup.find('script', type='application/json')
    d = json.loads(script_tag.string)
    conv = d['props']['pageProps']['serverResponse']['data']['linear_conversation']
    chat = []
    for turn in conv:
        _id = turn.get('id')
        message = turn.get('message')
        if message:
            author = message.get('author')
            role = author.get('role')
            content = message.get('content')
            if content:
                content_type = content.get('content_type')
                parts = content.get('parts')
                if parts is not None and content_type == 'text':
                    chat.append({'role': role, 'content': parts.pop()})
    return chat


def save_conversation(input_url):
    '''Saves the conversation to a JSON file'''
    conversation_id = input_url.split('/')[-1]
    soup = fetch_webpage_content(input_url)
    chat = get_conversation(soup)
    out_fname = f'oai-chat-{conversation_id}.json'
    print(f"Saving conversation to {out_fname}")
    with open(out_fname, 'w') as f:
        json.dump(chat, f, indent=4)

def main():
    '''Main function'''
    args = parse_args()
    if args.input_file:
        with open(args.input_file, 'r') as f:
            input_urls = f.readlines()
        input_urls = [x.strip() for x in input_urls]
        print(input_urls)
        for input_url in input_urls:
            time.sleep(1)
            print(f"Fetching conversation from {input_url}")
            save_conversation(input_url)
    elif not args.input_url:
        input_url = "https://chat.openai.com/share/4ad82157-c4b9-421e-9e33-7902ea940d71"
        save_conversation(input_url)
    else:
        save_conversation(args.input_url)


if __name__ == '__main__':
    main()
