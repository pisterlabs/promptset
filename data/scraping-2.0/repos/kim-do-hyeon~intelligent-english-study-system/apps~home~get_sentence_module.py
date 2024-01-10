
import sys
import ssl
import json
import openai
import urllib.request
from apps.home.api_key import *

ssl._create_default_https_context = ssl._create_unverified_context

def gen_sentence(word1, word2) :
    openai.api_key = get_chatgpt_api_key()
    prompt = "Please create a sentence with {} and {}.".format(word1, word2)
    model_engine = "text-davinci-002"

    ''' ChatGPT API '''
    completions = openai.Completion.create(
        engine=model_engine,    
        prompt=prompt,
        max_tokens=30,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = completions.choices[0].text
    sentence = (message.strip())

    ''' Papago API '''
    client_id, client_secret = get_papago_api_key()
    encText = urllib.parse.quote(sentence)
    data = "source=en&target=ko&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    response_body = ""
    if(rescode==200):
        response_body = response.read()
    else:
        response_body = "Error"
        print("Error Code:" + rescode)
    translate_dict = json.loads(response_body.decode('utf-8'))
    translate_data = (translate_dict['message']['result']['translatedText'])
    return sentence, translate_data