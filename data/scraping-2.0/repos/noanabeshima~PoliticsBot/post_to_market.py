import requests
import json
import time
import os

from bs4 import BeautifulSoup
from openai import OpenAI


client = OpenAI()


def query(prompt, messages, model, max_tokens=300, **kwargs):
        '''
        prompt: Or[list[dict[str, str]], str]
        returns a single string, the response from model
        '''
        if isinstance(prompt, str):
            prompt = [{'role': 'system', 'content': prompt}]


        for retry_attempt in range(1, 6):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages = prompt+messages,
                    max_tokens=max_tokens,
                    **kwargs
                )
                break
            except Exception as e:
                print(e)
                print('Model: ', model)
                print('kwargs: ', kwargs)
                print('prompt: ', str(prompt).replace('\n', '\\n'))

                if retry_attempt == 5:
                    raise e
                retry_interval = int(15*(retry_attempt**1.6))
                print(f'Sleeping for {retry_interval} seconds...')
                time.sleep(retry_interval)
                continue
        return response


def get_sfpost_content(link):
    res = requests.get(link)


    soup = BeautifulSoup(res.text, 'html.parser')


    content = soup.find('div', {'class': 'article-content'})

    article_text = ''
    for child in content.children:
        if 'article-content' in child.get('class', []):
            article_text += '\n\n'+child.text
    article_text = article_text.strip()

    return article_text


prompt_template = '''Given the article below, please output a manifold market via the create_market function provided, directly related to the article. You must call the create_market function, please don't just output a json directly. Each market should be carefully operationalized in the description and should be a binary market. The markets should be fairly easy to evaluate as YES or NO, evaluation shouldn't require lots of manual effort.

{article}'''

tools = [
    {
        'type': 'function',
        'function':
            {
                "name": "create_market",
                "description": "Create a Manifold Market.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "(required) A title for the market, also a question. For example, 'Will Joe Biden win the 2024 presidential election?'",
                        },
                        "market_description": {
                            "type": "string",
                            "description": "(required) A careful description of the question going into further details to clarify the exact operationalization of the market so that it is easy to see under what conditions the market would resolve Yes, No, and possibly N/A."
                        },
                        "close_time": {
                            "type": "string",
                            "description": "Market close time as string of form MONTH/DAY/YEAR. For example, 7/31/2024. Must close in the future."
                        }
                    }
                },
            },
        'required': ['title', 'market_description']
    }
]




post_template = '''{title}
Published {published}

{body}'''

def get_article(post):
    link = post['link']
    title = post['title']
    published = post['published']
    body = get_sfpost_content(link)

    post = post_template.format(title=title, published=published, body=body)

    return post

from datetime import datetime
import pytz
def close_time_to_iso(close_time):
    dt = datetime.strptime(close_time, "%m/%d/%Y")
    dt = dt.replace(tzinfo=pytz.UTC)
    return int(dt.timestamp() * 1000)

import requests

def get_market_kwargs(article, model='gpt-4',):
    for i in range(3):
        try:
            response = query(prompt=prompt_template.format(article=article),
                        messages=[],
                        model=model,
                        tools=tools,
                        temperature=0.8)


            tool_calls = response.choices[0].message.tool_calls
            market_kwargs = json.loads(tool_calls[0].function.arguments)
            break
        except Exception as e:
            print('sleeping')
            if i == 2:
                assert False, f'{e}\n\n Trouble creating market in get_market_kwargs'
            time.sleep(4)
            continue
        

    market_kwargs['close_time'] = close_time_to_iso(market_kwargs['close_time'])


    return market_kwargs

def create_market(title, market_description, close_time):
    api_key = os.environ.get('POLITICS_BOT_API_KEY')
    url = 'https://manifold.markets/api/v0/market'

    data = {
        'outcomeType': 'BINARY',
        'question': title,
        'descriptionMarkdown': market_description,
        'closeTime': close_time,
        'initialProb': 50
    }
    headers = {'Authorization': 'Key ' + api_key,
            'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, json=data)

    return response

def create_market_from_post(post):
    article = get_article(post)
    market_kwargs = get_market_kwargs(article)

    market_kwargs['market_description'] += f'\n\nThis market was automatically created from [{post["link"]}]({post["link"]}).'

    assert 'title' in market_kwargs
    assert 'market_description' in market_kwargs
    assert 'close_time' in market_kwargs
    create_market(**market_kwargs)
    return market_kwargs
