from bs4 import BeautifulSoup
import requests
import openai
import math
import json
import re
import os
import smtplib
import textwrap
import numpy as np
from time import time,sleep
from email.message import EmailMessage
from subprocess import PIPE, run
from django.conf import settings
from django.core.mail import send_mail
from .mqtt_2 import client as mqtt_2_client

openai.api_key = os.getenv("OPENAI_API_KEY")
file_name = "training_data.jsonl"

msg = EmailMessage()
msg.set_content("success!! Hello")
#********************************************************************************************************************
#********************************************************************************************************************
#********************************************************************************************************************
#********************************************************************************************************************
#********************************************************************************************************************
def gpt3_embedding(content, engine='text-similarity-ada-001'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def similarity(v1, v2):  # return dot product of two vectors
    return np.dot(v1, v2)


def search_index(text, data, count=20):
    vector = gpt3_embedding(text)
    scores = list()
    for i in data:
        score = similarity(vector, i['vector'])
        #print(score)
        scores.append({'content': i['content'], 'score': score})
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    return ordered[0:count]


def gpt3_completion(prompt, engine='text-davinci-002', temp=0, top_p=1.0, tokens=2000, freq_pen=0.25, pres_pen=0.0, stop=['<<END>>']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            with open('gpt3_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)
def getAnswers(userID, query):
    with open( userID + '-index.json', 'r') as infile:
        data = json.load(infile)
    if(len(data) == 0):
        answer = openai.Completion.create(
            model="text-davinci-003",
            prompt=query,
            max_tokens=1000, # Change amount of tokens for longer completion
            temperature=0
        )
        return answer.choices[0].text
    #print(data)
    #print(query)
    results = search_index(query, data)
    #print(results)
    #exit(0)
    answers = list()
    # answer the same question for all returned chunks
    for result in results:
        prompt = open_file('prompt_answer.txt').replace('<<PASSAGE>>', result['content']).replace('<<QUERY>>', query)
        answer = gpt3_completion(prompt)
        print('\n\n', answer)
        answers.append(answer)
    # summarize the answers together
    all_answers = '\n\n'.join(answers)
    chunks = textwrap.wrap(all_answers, 10000)
    final = list()
    for chunk in chunks:
        prompt = open_file('prompt_summary.txt').replace('<<SUMMARY>>', chunk)
        summary = gpt3_completion(prompt)
        final.append(summary)
    print('\n\n=========\n\n', '\n\n'.join(final))
    return '\n\n'.join(final)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def gpt3_embedding(content, engine='text-similarity-ada-001'):
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector

def makeJsonFile(content, userID):
    # pros = 10
    # rc, mid = mqtt_2_client.publish('django/progress/setting/'+userID, pros)
    alltext = content
    chunks = textwrap.wrap(alltext, 4000)
    result = list()
    for chunk in chunks:
        embedding = gpt3_embedding(chunk.encode(encoding='ASCII',errors='ignore').decode())
        info = {'content': chunk, 'vector': embedding}
        print(info, '\n\n\n')
        result.append(info)
    with open(userID + '-index.json', 'w') as outfile:
        json.dump(result, outfile, indent=2)
        
#********************************************************************************************************************
#********************************************************************************************************************
#********************************************************************************************************************
#********************************************************************************************************************
#********************************************************************************************************************
        
def get_questions(context):
    print('context-----------', context)
    print('aaaaaaa', "write questions as array fommat based on the text below\n\nText: \"" + context + "\"")

    questions = openai.Completion.create(
        model= "text-davinci-003",
        prompt= "write questions as array fommat based on the text below\n\nText: \"" + context + "\"",
        temperature= 0,
        max_tokens= 2000,
        top_p= 1,
        frequency_penalty= 0.5,
        presence_penalty= 0,
    )
    print("questions", questions.choices[0].text[questions.choices[0].text.find('['):questions.choices[0].text.find(']')+1])
    return questions.choices[0].text[questions.choices[0].text.find('['):questions.choices[0].text.find(']')+1]

def get_answers(context, questions):
    response = openai.Completion.create(
        model= "text-davinci-003",
        prompt= "write answers as  array for these questions based on the text below\n\nText: \"" + context + "\"\n\nQuestions:\n" + questions,
        temperature= 0,
        max_tokens= 2000,
        top_p= 1,
        frequency_penalty= 0.5,
        presence_penalty= 0,
    )
    print(response.choices[0].text[response.choices[0].text.find('['):response.choices[0].text.find(']')+1])
    return response.choices[0].text[response.choices[0].text.find('['):response.choices[0].text.find(']')+1]


def scrapFromURL(userID, url:str)->str:
    baseURL = url[0 : url.find('/', 8, 100)]
    print('BaseURL', baseURL)
    resultStr = ''

    page = requests.get(url)
    try:
        soup = BeautifulSoup(page.content, 'html.parser')
    except:
        pass
    aList = soup.find_all('a')
    i = 0
    resultStr += soup.get_text()
    print('related pages number:', len(aList))
    while i < len(aList):
        print("*******************", aList[i].get("href"))
        if(aList[i].get("href") == None):
            i += 1
            print('##',i)
            continue
        if aList[i].get("href")[0] == '/' or aList[i].get("href").find(baseURL) > 0:
            pros =10 + math.floor(70/len(aList)) * i
            print('##',i)
            rc, mid = mqtt_2_client.publish('django/progress/setting/'+userID, pros)
            print('scraping related url', baseURL + aList[i].get("href"))
            resultStr += BeautifulSoup(requests.get(baseURL + aList[i].get("href")).content, 'html.parser').get_text()
        i += 1
    pros = 90
    rc, mid = mqtt_2_client.publish('django/progress/setting/'+userID, pros)
    return resultStr
def createTrainData(questionArrayStr, answerArrayStr, userID):
    print("1")
    try:
        questionArray = json.loads(questionArrayStr)
        answerArray = json.loads(answerArrayStr)
        print("2")
    except:
        print("3")
        return
    resultArray = []
    print("4", questionArray, answerArray)
    for index in range(len(questionArray)):
        temp = {"prompt":"", "completion": ""}
        temp["prompt"] = questionArray[index] + "\n\n###\n\n"
        temp["completion"] = answerArray[index] + "\n"
        resultArray.append(temp)
    print("5", resultArray)

    with open(userID+"-"+file_name, "a") as output_file:
        for entry in resultArray:
            json.dump(entry, output_file)
            output_file.write("\n")
def out(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout
def main( doc_url, userID ):
    pros=0
    # print("111111111111111111111111111111111111")
    try:
        Tcontext = scrapFromURL(userID, doc_url)
    except Exception as e:
        print(e)
        return {"code": 500, "message": "failed", "body": "We can't get data from your url"}
    Tcontext = Tcontext.strip()
    Tcontext = "".join(re.findall("[a-zA-Z 0-9:!@#$%^&*?/,.<>\|';{}=+-_()]", Tcontext))
    print('-------------Tcontext------------', Tcontext)
    makeJsonFile(Tcontext, userID)
    email(userID, 'Training is completed!')
    pros = 100
    rc, mid = mqtt_2_client.publish('django/progress/setting/'+userID, pros)
    return {"code": 200, "message": "success", "body": True}

    ## progress
    # pros+=4
    # rc, mid = mqtt_2_client.publish('django/progress/setting/'+userID, pros)
    # print(pros, '_rc',rc, 'mid', mid)    
    # nToken = len(Tcontext.split())
    # print('number of tokens', nToken)
    # result = []
    # for i in range(math.ceil(len(Tcontext) / 1000)):
    #     context = Tcontext[i*1000:(i + 1)*1000]
    #     try:
    #         questionArrayStr = get_questions(context)
    #         answerArrayStr = get_answers(context, questionArrayStr)
    #         createTrainData(questionArrayStr, answerArrayStr, userID)
    #     except Exception as e:
    #         pass
    # ## progress
    # pros+=4
    # rc, mid = mqtt_2_client.publish('django/progress/setting/'+userID, pros)
    # print(pros, '_rc',rc, 'mid', mid)    
    # print("fine tune started!!!")
    # try:
    #     fineTuneConfirm = out("openai api fine_tunes.create -t " + userID + "-training_data.jsonl -m davinci")
    #     pros+=10
    #     rc, mid = mqtt_2_client.publish('django/progress/setting/'+userID, pros)
    #     print(pros, '_rc',rc, 'mid', mid)    
    #     ## progress
    # except:
    #     return {"code": 500, "message": "failed", "body": "please run \'pip install openai!\'"}
    # while fineTuneConfirm.find("succeeded") < 0:
    #     print("#########", fineTuneConfirm.split("\n"))
    #     fineTuneConfirm = out(fineTuneConfirm.split("\n")[-3])
    #     if pros<90:
    #         pros+=10
    #         rc, mid = mqtt_2_client.publish('django/progress/setting/'+userID, pros)
    #         print(pros, '_rc',rc, 'mid', mid)  
    # ## 100%
    # pros=100
    # rc, mid = mqtt_2_client.publish('django/progress/setting/'+userID, pros)
    # print(pros, '_rc',rc, 'mid', mid)  
    # print("***********", fineTuneConfirm.split("\n"), "=======", fineTuneConfirm.split("\n")[-2])
    return {"code": 200, "message": "success", "body": fineTuneConfirm.split("\n")[-2].split(" ")[4]}

def completion(userID, prompt):
    # print("modelID:", modelID)
    # if modelID == "":
    #     modelID = "text-davinci-003"
    # try:
    #     answer = openai.Completion.create(
    #         model=modelID,
    #         prompt=prompt,
    #         # top_p=1,
    #         max_tokens=256, # Change amount of tokens for longer completion
    #         temperature=0
    #     )
    # except Exception as ex:
    #     return {"code": 500, "message": "failed", "body": type(ex).__name__}
    # print("answer---------", answer.choices[0].text)
    # if answer.choices[0].text.find("###") > 0:
    #     return {"code": 200, "message": "success", "body": answer.choices[0].text.split("###")[1].strip().split("\n")[0]}
    # return {"code": 200, "message": "success", "body": answer.choices[0].text}

    
    return {"code": 200, "message": "success", "body": getAnswers(userID, prompt)}

def email(emailAddress, contentText):
    subject = 'SmartResponse'
    # message = ' Fine-tune is completed. Test your model. '
    message = contentText
    email_from = settings.EMAIL_HOST_USER
    recipient_list = [emailAddress]
    send_mail( subject, message, email_from, recipient_list )
    return
# main("")
