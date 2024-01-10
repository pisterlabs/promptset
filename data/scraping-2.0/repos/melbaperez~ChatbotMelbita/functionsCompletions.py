import os
import openai
import time
import logging
from utils import *

openai.organization = os.environ["OPENAI_ORGANIZATION_KEY"] 
openai.api_key = os.environ["OPENAI_API_KEY"]

contextChatbot = readYaml("data/prompts.yaml")['PROMPT_CHATBOT']
logging.basicConfig(filename='log/app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', encoding="utf-8")


def getCompletion(prompt, numberTry = 0, model="text-davinci-003"):
    ''' GPT response to classify user input. '''
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=0.6,
            max_tokens=7,
            presence_penalty=0.6
        )
        answer = response.choices[0].text.strip()
        return answer
    except Exception as e:
        logging.warning(str(numberTry) + ". " + str(e))
        if numberTry >= 2:
            return "Ahora mismo me encuentro sobrecargada, vuelva a intentarlo dentro de unos segundos."
        else:
            time.sleep(10)
            getCompletion(prompt, numberTry = ++numberTry)


def completion(conversation, sentencesDatabase, numberTry = 0, model="text-davinci-003"):
    ''' GPT response obtaining function focused on continuing conversation. '''
    try:
        prompt = f"{contextChatbot.format(getMessageTime())}\nPrimera sección:\n{sentencesDatabase}\nSegunda sección:\n{conversation}"
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=0.9,
            max_tokens=300,
            presence_penalty=0.6,
            stop=[" Usuario:", " AI:"]
        )
        answer = response.choices[0].text.strip()
        return answer
    except Exception as e:
        logging.warning(str(numberTry) + ". " + str(e))
        if numberTry >= 2:
            return "Ahora mismo me encuentro sobrecargada, vuelva a intentarlo dentro de unos segundos."
        else:
            time.sleep(10)
            completion(prompt, numberTry = ++numberTry)


def getCategorySentence(text, numberTry = 0):
    ''' Detects whether a phrase is a command, contains information or is a statement. '''
    prompt = readYaml("data/prompts.yaml")['PROMPT_CLASSIFICATION'].format(text)
    response = getCompletion(prompt, 0)
    logging.info(f"{text} --> {response}")
    return response
