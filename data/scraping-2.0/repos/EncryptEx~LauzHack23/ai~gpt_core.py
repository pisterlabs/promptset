from time import sleep
from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()


client = OpenAI()

majorhistory = []
limit = 16000 # depends on the model

initialPrompt = """I'll give a part of a log file and I'll ask you a question. 
                    Just answer 'OK' unless the question starts with 'EOF'. 
                    After the partial log piece (when you recieved the 'EOF') I'll ask a question, an example input = (in the log file) Nov 27 22:52:56 exampleuser sudo:    jonny : TTY=pts/0 ; PWD=/home/jonny ; USER=root ; COMMAND=/usr/bin/apt purge google-chrome-stable, did the user jonny delete any program; 
                    output = SUCCESS: the user jonny deleted google chrome on November 27 22:52:56. 
                    If the question cannot be strictly answered by the information given, just answer 'INSUFFIEICENT_INFO' else, return SUCCESS at the beggining (and continued by your answer)."""

messag = None #"INSUFFICIENT_INFO" Set to none each time you want to start a new conversation
history_bot = None
history_user = None
def analizeLog(prompt):
    global messag, history_bot, history_user
    if(messag == None):
        messag=[{"role": "system", "content": "You are a chatbot"}]
    
        ## build a chat history: you can CONDITION the bot on the style of replies you want to see - also getting weird behaviors... such as KanyeGPT
        history_bot = ["Yes, I'm ready! Please provide the log to scan."]
    
        # ask ChatGPT to return STRUCTURED, parsable answers that you can extract easily - often better providing examples of desired behavior (1-2 example often enough)
        history_user = [initialPrompt]
        # Remember that you should answer 'OK' unless you see EOF in the start of the prompt, that means that question is ready to be answered since the log file has been already sent by its entirety
    

        for user_message, bot_message in zip(history_user, history_bot):
            messag.append({"role": "user", "content": str(user_message)})
            messag.append({"role": "system", "content": str(bot_message)})
    messag.append({"role": "user", "content": str(prompt)})
    
    response = client.chat.completions.create(model="gpt-3.5-turbo-16k",
        messages=messag)
    result = ''
    for choice in response.choices:
        result += choice.message.content
   
    messag.append({"role": "system", "content": str(result)})
    majorhistory.append(messag)

    return result




def askGPT(filePath, question)-> str:
    aux = question+"\n "
    isFirsttime = True
    iCounter = 0
    aux = ""
    answers = []
    with open(filePath, "r") as f:
        aux = f.read()
        while len(aux) >= limit-len(str(messag)):
            softRestartConversation()
            analizeLog(aux[:limit])
            aux = aux[limit:]
            sleep(1)
            answers.append(analizeLog("EOF From the previous sent log, "+question))

    if aux:
        analizeLog(aux)
    
    
    for answer in answers:
        if answer.startswith("SUCCESS"):
            return answer
    return "There is not enought information, try with another log"


def softRestartConversation():
    global messag, history_bot, history_user
    messag = None
    history_bot = None
    history_user = None
    print("Conversation restarted")

def hardRestartConversation():
    global majorhistory
    
    majorhistory = None
    softRestartConversation()

    print("Conversation HARDrestarted")

client.api_key = os.getenv("OPENAI_API_KEY")

# fp = "/home/jaume/Desktop/CODE/LauzHack23/bot/examples/auth.log"
# question = "Did the user jonny delete any program?"

# result = askGPT(fp, question)

def askAnotherQuestion(question):
    answers = []
    for i in majorhistory:
        i.append({"role": "user", "content": str(question)})
        response = client.chat.completions.create(model="gpt-3.5-turbo-16k",
        messages=messag)
        result = ''
        for choice in response.choices:
            result += choice.message.content
    
        i.append({"role": "system", "content": str(result)})
        answers.append(result)

    for answer in answers:
        if answer.startswith("SUCCESS"):
            return answer
        else: 
            return "There is not enough information, try with another question"
    

