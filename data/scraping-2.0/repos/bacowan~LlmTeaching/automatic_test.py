import openai
import os
from config import openapi_key, model
from frameworkModel import FrameworkModel
from rawModel import RawModel
import logging
import random
from responder import ask_further_question

random.seed(0)

item = input("Which file to test? ")
problem = open("./problems/" + item + ".txt").read()
code = open("./code/" + item + ".txt").read()
question = open("./questions/" + item + ".txt").read()

student_type = input("Should the suggested questions be 1. for a student on the right track, or 2. for a student who feels lost? ")
if student_type == "1":
    logname = 'chat_logs_' + item + '_1.log'
    role = "on the right track, but need a little nudge"
elif student_type == "2":
    logname = 'chat_logs_' + item + '_2.log'
    role = "really stuck"
else:
    logname = 'chat_logs_' + item + '_3.log'
    role = None

logging.basicConfig(filename=logname, encoding='utf-8', level=logging.INFO)

print("Item: " + item)
print("Problem: " + problem)
print("Code: " + code)
print("Question: " + question)
  
openai.api_key = openapi_key

logging.info("Problem: " + item)
logging.info("VisibleQuestion: " + question)

model = FrameworkModel()
response = model.start_conversation(problem, question, code)
logging.info("VisibleResponse: " + response)
print(response)

for _ in range(5):
    retry = True
    while retry == True:
        try:
            newPrompt = ""
            while newPrompt == "":
                try:
                    prompt = ask_further_question(model.conversation, role)
                    logging.info("SuggestedQuestion: " + prompt)
                    print("Suggested prompt:")
                    print(prompt)
                    newPrompt = input("New prompt: ")
                except:
                    print("oops; trying again")
            logging.info("VisibleQuestion: " + newPrompt)
            response = model.send_prompt(newPrompt)
            logging.info("VisibleResponse: " + response)
            print(response)
            retry = False
        except:
            print("oops; trying again")
            retry = True