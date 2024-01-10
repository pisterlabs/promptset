#!/usr/bin/env python
# coding: utf-8

# In[89]:


import os
import requests
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import tool
from bs4 import BeautifulSoup
from datetime import date
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate


# In[90]:


def get_today():
    today = date.today()
    return f'{today.month:02}-{today.day:02}-{today.year}'
    #return f'{today.month:02}-{today.day+1}-{today.year}'

def get_url():
    return f'https://fremontunified.org/horner/news/daily-bulletin-{get_today()}/'


# In[91]:


@tool
def fetch_horner_website(text: str) -> str:
    '''Returns the contents of the "Daily Update" contents for Horner Middle School.
    Expects an input of a empty string '' and returns the contents of the webpage or a string "No Updates" for that day.
    '''
    headers = {
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "max-age=0",
        "sec-ch-ua": "\"Chromium\";v=\"116\", \"Not)A;Brand\";v=\"24\", \"Google Chrome\";v=\"116\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"macOS\"",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "none",
        "sec-fetch-user": "?1",
        "sec-gpc": "1",
        "upgrade-insecure-requests": "1"
      }
    url = get_url()
    
    response = requests.get(url,headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text().replace('\n','')
    if text.startswith('Page Not Found'):
        html_content = f'No updates for {get_today()}!'
    else:
        html_content = text
    return html_content


# In[92]:


def run_agent():
    openai_api_key = os.environ['OPENAI_API_KEY'] 
    print(f'open API key: {openai_api_key}')
    llm = ChatOpenAI(temperature=0,model='gpt-4',openai_api_key=openai_api_key)
    # what tools are available to agent
    tools = load_tools(tool_names= [],llm=llm)
    tools += [fetch_horner_website]
    # ZERO_SHOT_REACT_DESCRIPTION - opposite of few shot. i.e. no examples will be given to the LLM. directly ask something.
    agent = initialize_agent(
        tools, 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    ## now build the prompt template
    system_prompt = SystemMessagePromptTemplate.from_template('''You are a text summarizing service for Horner Middle School. 
    When invoked, you will fetch the Horner website, extract the content and then summarize the contents. You will skip the summarization if the website 
    has no updates. This is indicated by the string "No Updates" followed by the date.''')    
    human_prompt = HumanMessagePromptTemplate.from_template('''Please fetch the latest from Horner Updates Website and condense the message so that it is easy to read. ''')    
    chat_template = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    
    result = agent.run(chat_template.format())
    return result


# In[100]:


def send_message_to_telegram(message,url=get_url()):
    bot_token = os.environ.get('BOT_TOKEN') or print('No Telegram bot token found!')
    chat_id = os.environ.get('CHAT_ID') or print('No Telegram chat_id found!')
    # Send scraped webpage content to Telegram channel
    telegram_api_url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    message_text = f"Scraped webpage content:\n\n{message}\n\nFull notification: {url}"

    params = {
        'chat_id': chat_id,
        'text': message_text,
        'parse_mode': 'HTML'
    }

    response = requests.post(telegram_api_url, params=params)
    if response.status_code == 200:
        print("Message sent successfully to Telegram channel!")
    else:
        print(f"{response.status_code}: Failed to send message to Telegram channel.")
        print(response.json())



# In[94]:


result = run_agent()


# In[101]:


send_message_to_telegram(result)





