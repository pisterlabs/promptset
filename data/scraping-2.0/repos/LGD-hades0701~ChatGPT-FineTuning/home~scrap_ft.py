from bs4 import BeautifulSoup
import requests
import openai
import math
import json
import re
from subprocess import PIPE, run
openai.api_key = "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"
file_name = "training_data.jsonl"

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


def scrapFromURL(url:str)->str:
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
            continue
        if aList[i].get("href")[0] == '/' or aList[i].get("href").find(baseURL) > 0:
            print('scraping related url', baseURL + aList[i].get("href"))
            resultStr += BeautifulSoup(requests.get(baseURL + aList[i].get("href")).content, 'html.parser').get_text() + "-"*77
        i += 1
    return resultStr
def createTrainData(questionArrayStr, answerArrayStr, userID):
    try:
        questionArray = json.loads(questionArrayStr)
        answerArray = json.loads(answerArrayStr)
    except:
        return
    resultArray = []
    for index in range(len(questionArray)):
        temp = {"prompt":"", "completion": ""}
        temp["prompt"] = questionArray[index] + "\n\n###\n\n"
        temp["completion"] = answerArray[index] + "\n"
        resultArray.append(temp)
    with open(userID+"-"+file_name, "a") as output_file:
        for entry in resultArray:
            json.dump(entry, output_file)
            output_file.write("\n")
def out(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout
def main( doc_url, userID ):
    

    Tcontext = scrapFromURL(doc_url)
    Tcontext = Tcontext.strip()
    Tcontext = "".join(re.findall("[a-zA-Z 0-9:!@#$%^&*?/,.<>\|';{}=+-_()]", Tcontext))
    print('-------------Tcontext------------', Tcontext)
    nToken = len(Tcontext.split())
    print('number of tokens', nToken)
    result = []
    for i in range(math.ceil(len(Tcontext) / 1000)):
        context = Tcontext[i*1000:(i + 1)*1000]
        questionArrayStr = get_questions(context)
        answerArrayStr = get_answers(context, questionArrayStr)
        createTrainData(questionArrayStr, answerArrayStr, userID)
    fineTuneConfirm = out("openai api fine_tunes.create -t " + userID + "-training_data.jsonl -m davinci")
    print("#########", fineTuneConfirm.split("\n"))
    print("#########", "openai api fine_tunes.create -t " + userID + "-training_data.jsonl -m davinci")
    # print(fineTuneConfirm.split("\n"), fineTuneConfirm.split("\n")[-3])
    fineTuneConfirm1 = out(fineTuneConfirm.split("\n")[-3])
    print("#########", fineTuneConfirm1.split("\n"))

    fineTunedModel = out(fineTuneConfirm1.split("\n")[-3])
    print("***********", fineTunedModel.split("\n"), "=======", fineTunedModel.split("\n")[-4])
    return fineTunedModel.split("\n")[-4].split(":")[-2] + ":" + fineTunedModel.split("\n")[-4].split(":")[-1]
def completion(modelID, prompt):
    if modelID == "":
        modelID = "text-davinci-003"
    answer = openai.Completion.create(
        model=modelID,
        prompt=prompt,
        max_tokens=1000, # Change amount of tokens for longer completion
        temperature=0
    )
    return answer.choices[0].text
# main("")
