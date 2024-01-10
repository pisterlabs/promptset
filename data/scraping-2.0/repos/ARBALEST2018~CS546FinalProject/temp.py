import csv
import configs
import openai
import os
import event
import argument
import nltk
import json
import string
from collections.abc import Iterable
from event_format import event_format
from event_template import event_template
from load_events import eventContainer
from predictions import predictionContainer

openai.api_key = configs.OPENAI_API_KEY


# def extract_event(sentence):
#     temp = {"user": sentence+" "+configs.EVENT_EXT}
#     print(ask_openai(temp))
    
# def extract_arguments(sentence):
#     temp = {"user": sentence+" "+configs.ARGUMENT_EXT}
#     temp = ask_openai(temp)
#     # print(temp)
#     toReturn = []
#     for line in temp["content"].split("\n"):
#         words = tokenizer.tokenize(line)
#         toAppend = ""
#         for word in words:
#             if not word.isnumeric():
#                 toAppend+=word+" "
#         toReturn.append(toAppend.strip())
#     return toReturn
        

# def loadjsonlines(inputfilename,outputwriter = None, process = None):
#     f = open(inputfilename)
#     for jsonline in open(inputfilename):
#         temp = json.loads(jsonline)
#         if (process != None):
#             temp = process(temp)
#         if (outputwriter != None):
#             outputwriter(temp)
        
# def process(input):
#     toReturn = []
#     words = []
#     sentences = []
#     for sentence in input["sentences"]:
#         temp = ""
#         first = True
#         for word in sentence:
#             words.append(word)
#             if word in string.punctuation or first:
#                 temp += word
#             else:
#                 temp += (" "+word)
#             first = False
        
#         sentences.append(temp)
#     triplets = []
#     for triplet in input["gold_evt_links"]:
#         tri = ""
#         for i in range(0,2):
#             temp = ""
#             for j in range(triplet[i][0], triplet[i][1]+1):
#                 temp += words[j]
#                 temp += " "
#             tri += temp + " : "
#         tri += triplet[2]
#         triplets.append(tri)
#     return [sentences,triplets]
    
# n = "test"

# w = open("new_data/"+n+".txt", mode='w', encoding='utf-8')    

# def outwriter(input):
#     if isinstance(input, list):
#         for item in input:
#             outwriter(item)
#         w.write("\n")
#     else:
#         w.write(input+"\n")

# loadjsonlines("rams/data/"+n+".jsonlines", process=process, outputwriter=outwriter)






# n = "dev"

# jsonlinesToOpen = "rams/data/"+n+".jsonlines"
# w = open("new_data/"+n+".txt", mode='w', encoding='utf-8')
# with open(jsonlinesToOpen) as f:
#     i = 0
#     for line in f:
#         jsonline = json.loads(line)
#         if i < 2:
#             for sentence in jsonline["sentences"]:
#                 temp = ""
#                 first = True
#                 for word in sentence:
#                     temp += word+" "
#                 w.write(temp[:-1]+"\n")
#             w.write("\n")
# messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Who won the world series in 2020?"},
#     {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#     {"role": "user", "content": "Where was it played?"}
#   ]

# def makePrompt(eventTemplate, trainEventContainer, testEvent, max_num, with_location = True):
#     branch = True
#     # toReturn = ""
#     message = []
#     toReturn = []
#     if (branch):
#         template = eventTemplate.getEventTemplateByIndex(testEvent["i-label"])
#         if (template == None):
#             return None
        
#         s = {"role":"system"}
#         s["content"] = configs.INSTRUCTION
#         toReturn.append(s)
        
#         text = ""
#         for word in testEvent["text"]:
#             text += word+" "
#         text = configs.DOCUMENT_HEAD+"\n"+text[:-1]+"\n"
#         text += "\n"
        
#         text += configs.TEMPLATE_HEAD+"\n"
#         text += template["template"]+"\n"
#         text += "\n"
        
#         text += configs.TASK_HEAD+"\n"
#         text += configs.TASK+"\n"
#         for role in template["roles"]:
#             text += role+"\n"
            
#         u = {"role":"user"}
#         u["content"] = text
#         toReturn.append(u)
        
        
        
#         # toReturn += configs.INSTRUCTION_HEAD+"\n"
#         # toReturn += configs.INSTRUCTION+"\n"
#         # toReturn += "\n"
        
#         # toReturn += configs.DOCUMENT_HEAD+"\n"
#         # text = ""
#         # for word in testEvent["text"]:
#         #     text += word+" "
#         # toReturn += text[:-1]+"\n"
#         # toReturn += "\n"
        
#         # toReturn += configs.TEMPLATE_HEAD+"\n"
#         # toReturn += template["template"]+"\n"
#         # toReturn += "\n"
        
#         # toReturn += configs.TASK_HEAD+"\n"
#         # toReturn += configs.TASK+"\n"
#         # for role in template["roles"]:
#         #     toReturn += role+"\n"
        
#     # else:
#     #     if with_location:
#     #         toReturn += configs.PROMPT_0+"\n"+"\n"
#     #     else:
#     #         toReturn += configs.PROMPT_0_NL+"\n"+"\n"
#     #     trainEvents = trainEventContainer.getEventsById(testEvent["i-label"], [role[0] for role in testEvent["roles"]])
#     #     if (len(trainEvents)) == 0:
#     #         return ""
#     #     i = 0
#     #     while i < len(trainEvents) and i < max_num:
#     #         trainEvent = trainEvents[i]
            
#     #         text = ""
#     #         for word in trainEvent["text"]:
#     #             text += word+" "
#     #         text = configs.PROMPT_TEXT+str(i+1)+" : "+text[:-1]+"\n"
            
#     #         for role in trainEvent["roles"]:
#     #             toAdd = role[0]+" : "+role[1]
#     #             if with_location:
#     #                 toAdd += " ("+str(role[2])+")"
#     #             text += toAdd+"\n"
#     #         text += "##\n"
#     #         toReturn += text
#     #         i += 1
#     #     if with_location:
#     #         toReturn += "\n"+configs.PROMPT_1+"\n"+"\n"
#     #     else:
#     #         toReturn += "\n"+configs.PROMPT_1_NL+"\n"+"\n"
#     #     text = ""
#     #     for word in testEvent["text"]:
#     #         text += word+" "
#     #     toReturn += configs.PROMPT_TEXT+str(i+1)+" : "+text[:-1]+"\n"
#     #     for role in testEvent["roles"]:
#     #         toReturn += role[0]+" : \n"
#     #     if with_location:
#     #         toReturn += configs.PROMPT_EVENT_ARGUMENT_LOCATION+str(i+1)+" : "+"\n"
    
#     return toReturn


# # Instruction
# The following examples is how you should perform a multi-sentence information extraction tasks.
# Your tasks is to fill the arguments with entities you extract from the user input documents.
# You can use <UNK> to indicate the unknown entity.
# Answer the arguments in a consistent style as the example.

# # Example
# ## Document
# Transportation officials are urging carpool and teleworking as options to combat an expected flood of drivers on the road.
# ( Paul Duggan)
# -- A Baltimore prosecutor accused a police detective of “ sabotaging ” investigations related to the death of Freddie Gray, accusing him of fabricating notes to suggest that the state ’s medical examiner believed the manner of death was an accident rather than a homicide.
# The heated exchange came in the chaotic sixth day of the trial of Baltimore Officer Caesar Goodson Jr., who drove the police van in which Gray suffered a fatal spine injury in 2015.
# ( Derek Hawkins and Lynh Bui)

# ## Answer
# <arg1> killer(person/organisation/country): Officer Caesar Goodson Jr.
# <arg2> victim(person): Freddie Gray
# <arg3> instrument(vehicles/explosives/sharps/blunts/firearms/chemicals): <UNK>
# <arg4> place:Baltimore

def makePrompt(eventTemplate, trainEventContainer, testEvent, max_num, with_instruction = True):
    toReturn = []
    
    # template = eventTemplate.getEventTemplateByIndex(testEvent["i-label"])
    # if (template == None):
    #     return None
    
    if (with_instruction):
        s = {"role":"system"}
        
        temps = configs.INSTRUCTION_HEAD+"\n"
        temps += configs.INSTRUCTION_SAMPLE+"\n"
        temps += "\n"
        trainEvents = trainEventContainer.getEventsById(testEvent["i-label"], [role[0] for role in testEvent["roles"]])
        i = 0
        while i < len(trainEvents) and i < max_num:
            temps += configs.EXAMPLE_HEAD+" "+str(i+1)+"\n"
            trainEvent = trainEvents[i]
            temps += configs.EXAMPLE_DOCUMENT_HEAD+"\n"
            text = ""
            for word in trainEvent["text"]:
                text += word+" "
            temps += text[:-1]+"\n"
            temps += configs.EXAMPLE_ANSWER_HEAD+"\n"
            for role in trainEvent["roles"]:
                toAdd = role[0]+" : "+role[1]
                temps += toAdd+"\n"
            temps += "\n"
            i += 1
        s["content"] = temps
        # print(temps)
        toReturn.append(s)
    
    text = ""
    for word in testEvent["text"]:
        text += word+" "
    text = configs.DOCUMENT_HEAD+"\n"+text[:-1]+"\n"
    text += "\n"
    
    # text += configs.TEMPLATE_HEAD+"\n"
    # text += template["template"]+"\n"
    # text += "\n"
    
    text += configs.TASK_HEAD+"\n"
    text += configs.TASK+"\n"
    # for role in template["roles"]:
    #     text += role+"\n"
    for role in testEvent["roles"]:
        text += role[0]+" : \n"
        
    # print(text)
    
    u = {"role":"user"}
    u["content"] = text
    toReturn.append(u)
    
    return toReturn
        
        
eventFormats = event_format()
trainEventContainer = eventContainer("rams//data/train.jsonlines",eventFormats=eventFormats)
eventTemplates = event_template()
# json.dump(trainEventContainer.eventsAll, open("train_events.json", mode='w'), indent=4)
# eventFormats.updateFile()
testEventContainer = eventContainer("rams//data/test.jsonlines")
# predictions = predictionContainer("new_data/test-sample-gpt4.json")
# json.dump(testEventContainer.eventsAll, open("test_events.json", mode='w'), indent=4)
# predictionToWrite = open("new_data/test-pred.txt",mode='w',encoding='utf-8')
# outputToWrite = open("new_data/output.txt",mode='w',encoding='utf-8')

counter = 0
for num in [1,5]:
    predictions = predictionContainer("new_data/test-sample-"+str(num)+"-gpt4.json")  
    for testEvent in testEventContainer.eventsAll:
        current = predictions.getPredictionByDocKey(testEvent["doc_key"])
        prompt = makePrompt(eventTemplates, trainEventContainer, testEvent, num, True)
        if prompt == None or len(prompt) == 0 or len(current) > 2:
            continue
        print(json.dumps(prompt, indent=4))
        print()
        # text = ""
        # for word in testEvent["text"]:
        #     if (word != "."):
        #         text += word+" "
        #     else:
        #         text += word+"\n"
        
        current+=prompt
        response = openai.ChatCompletion.create(
            model=configs.OPENAI_CHAT_MODEL,
            response_format={ "type": "json_object" },
            messages=prompt,
        )
        current.append(response["choices"][0]["message"])
        predictions.updatePrediction(testEvent["doc_key"], current)
        predictions.updateFile()
        responseJSON = json.loads(response["choices"][0]["message"]["content"])
        print(json.dumps(responseJSON, indent=4))
        counter += 1
        
        # if (counter >= 10):
        #     break
        
        
        # prompt = makePrompt(eventTemplates, trainEventContainer, testEvent, 8, False)
        # if prompt == None or len(prompt) == 0:
        #     continue
        # print()
        # outputToWrite.write("\n")
        # print(prompt)
        # outputToWrite.write(prompt)
        # print()
        # outputToWrite.write("\n")
        # text = ""
        # for word in testEvent["text"]:
        #     if (word != "."):
        #         text += word+" "
        #     else:
        #         text += word+"\n"
        # predictionToWrite.write(text+"\n")
        # response = openai.Completion.create(
        #     model=configs.OPENAI_CHAT_MODEL,
        #     response_format={ "type": "json_object" },
        #     prompt=prompt,
        # )
        # predictionToWrite.write(response["choices"][0]["text"]+"\n\n")
        # print(response)
        # outputToWrite.write(response["choices"][0]["text"])
        # print()
        # outputToWrite.write("\n\n\n")
        # counter += 1
        
        # if (counter >= 10):
        #     break
    # predictionToWrite.close()
    # outputToWrite.close()
    # predictions.updateFile()

















