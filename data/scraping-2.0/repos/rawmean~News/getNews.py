import openai
import dotenv
from os import environ
import time

env_file = '../.env'
dotenv.load_dotenv(env_file, override=True)
openai.api_key = environ.get('OPENAI_API_KEY')
ec_uname = environ.get("UNAME")
ec_pass = environ.get("EC_PASS")
ec_url = environ.get("EC_URL")

print(f'url = {ec_url}')
from bs4 import BeautifulSoup as bs 
import requests 
 
payload = { 
	"uname": ec_uname, 
	"pass": ec_pass 
} 

def get_response_chat(messages) -> str:
    response = openai.ChatCompletion.create(
        # model=model, temperature=0.0, messages=messages
        model="gpt-3.5-turbo", temperature=0.01, messages=messages
    )
    return response["choices"][0]["message"]["content"] 


def refresh_news():
    s = requests.session() 
    response = s.post(ec_url, data=payload) 
     
    from bs4 import BeautifulSoup 
    soup = BeautifulSoup(response.content, "html.parser") 

    body = soup.body.text
    article = (body.split(sep='stories that matter')[1]).split(sep='Word of the day')[0]

    article = article.replace('“', '`')
    article = article.replace('”', '`')

    print(f'========= article: {article}')

    system_pre = "you are a helpful assistant"
    question_pre  = f'''
    First separate the following english text into paragraphs based on the topic. 
    The output must be in pure json and nothing else. 
    Do not include and paragraph that is about daily quiz.
    Do not include any paragraph that is not news.
    Limit hte number of paragraphs to 5. There shall be at most 5 paragraphs.
    Each paragraph must be converted to simple English.

    ### ENGLISH TEXT:
    {article}


    ### YOU MUST USE THIS JSON TEMPLATE:

    #####
    {{
        paragraphs:
            [
                PARAGRPAH_BASED_ON_TOPIC,
                ...
            ]
    }}
    #####
    '''

    messages_pre = [
        {"role": "system", "content": system_pre},
        {"role": "user", "content": question_pre},
    ]

    print('Starting the pre-query')
    paragraphs = get_response_chat(messages=messages_pre)

    print(f'paragraphs: {paragraphs}')


    system = '''
    You are an expert translator. You translate English text to very simple Spanish text that can be understood by a 10 year old. Do not use difficult spanish words.
    '''

    question = f'''
    Then translate each paragraph in the input Paragraphs given below to simple Spanish that can be understood by a 10 yeard old.
    Your response must be in the form of json. do not respond with anything but json format. Also, create a very short title for each of the news paragraphs.
    Replace all instances of character """ in the values in the json response with character "`""
    Do not use the character """ anywhere in the json response values.

    ### Input Paragraphs:
    {paragraphs}


    ### YOU MUST USE THIS JSON TEMPLATE:

    #####
    {{
        translations:
            [
                {{
                    english_version: PARAGRPAH_IN_ENGLISH,
                    english_title: TITLE_IN_ENGLISH,
                    translated_version: TRANSLATED_PARAGRAPH, // do not include any " character here
                    translated_title: TITLE_IN_ENGLISH
                }},
                ...
            ]
    }}
    #####

    '''

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    print('Starting the query ')
    translations = get_response_chat(messages=messages)

    print(translations)
    print(f"translation length = {len(translations)}")

    import json
    file_path = "./news.json"
    with open(file_path, "w") as json_file:
        json.dump(translations.replace('\n', ''), json_file)


    from subprocess import call
    import time
    import os

    # os.chdir(data_path)
    print('uploading to github')
    # call(['pwd'],shell=True)
    call(['git', 'add', '*.json'],shell=True)
    call(['git', 'commit', '-am', 'update' + str(time.time())],shell=True)
    call(['git', 'push', '-f', 'origin', 'master'],shell=True)


if __name__ == '__main__':
    while True:
        refresh_news()
        time.sleep(8 * 3600)