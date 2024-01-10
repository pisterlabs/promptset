import os
import pymongo
import json
import random
import time
import requests

import openai


openai.api_key = "thisSpaceForRent"



def getcuratedprojectplan(prompt):
    prempt = "Can you give me a project plan for the following project requirements, optionally with an estimated time and cost. here are the requirements and constraints: "
    
    prempt += "\n\n"
    
    prempt += prompt

    prempt += "\n"

    prempt += "Please keep the answer at a high level and brief, as if you are communicating with a non - technical person in an executive role. Do not provide time or cost estimates if not applicable or if the requirements and constraints are very generic or vague. "
    prempt += "\n"


    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a chatbot"},
                {"role": "user", "content": prempt},
            ]
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    print(result)
    
    print ("******************")
    
    ##process
    
    rjson = {}
    
    
    rdata = []
    
    
    rjson['data'] = rdata
        
    
    return result, rjson






def getprojectplan1(prompt, col):
    prempt = "Imagine you are an expert project manager. Following is a list of questions that should be asked by a sales rep to a potential client before providing a project plan and scope of work. "
    prempt += "\n Beginning of canonical list of questions: \n"

    for x in col.find():
        prempt += x['text']
        prempt += "\n"
    
    prempt += "\n End of canonical list of questions. \n"
    prempt += "Now here is a transcript of what the client was asked by the sales rep: "

    prempt += "\n"
    
    prempt += prompt

    prempt += "\n\n"

    prempt += "Please determine if all the questions have been asked by comparing with the canonical list of questions, and whether any are missing. If possible, provide some additional information which could be gathered from the client. If possible, also provide a draft project proposal given the transcript provided."
    prempt += "\n"
    prempt += "If there was a missing question or criteria please keep the answer brief and just mention the missing parts or questions. Primarily just compare the list of questions and if there are any missing, ONLY answer with the missing question."
    prempt += "\n If your response needs further questions please end the response with the text MISSING QUESTIONS all in capital letters, and if the response contains a solution end the response with SOLUTION PROVIDED also in all capital letters."



    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a chatbot"},
                {"role": "user", "content": prempt},
            ]
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    print(result)
    
    print ("******************")
    
    ##process
    
    rjson = {}
    
    
    rdata = []
    
    
    rjson['data'] = rdata
        
    
    return result, rjson


def getprojectplan2(prompt, col, col2):
    prempt = "Imagine you are an expert project manager. Following is a list of questions that should be asked before providing a project plan and scope of work. "
    prempt += "\n"

    for x in col.find():
        prempt += x['text']
        prempt += "\n"

    prempt += "Additionally, following are a set of general project parameters which should be considered before providing a plan. Some of them may not be relevant in this case. "
    prempt += "\n"

    for x in col2.find():
        prempt += x['name']
        prempt += "\n"

    prempt += "Now here is a transcript of what the client was asked"

    prempt += "\n"
    
    prempt += prompt

    prempt += "\n\n"

    prempt += "Please determine if all the criteria have been covered, and whether any are missing. Then provide a detailed project plan with a timeline and cost estimate given the parameters and the transcript provided."

    prempt += "\n\n"


    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a chatbot trained to be an expert project manager and estimator"},
                {"role": "user", "content": prempt},
            ]
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    print(result)
    
    print ("******************")
    
    ##process
    
    rjson = {}
    
    
    rdata = []
    
    
    rjson['data'] = rdata
        
    
    return result, rjson





def getprojectplan3(prompt, col):
    prempt = "Imagine you are an expert project manager."

    prempt += "Following are a set of general project parameters which should be considered before providing a plan. Some of them may not be relevant in this case. "
    prempt += "\n"

    for x in col.find():
        prempt += x['name']
        prempt += "\n"

    prempt += "Now here is a transcript of what the client was asked"

    prempt += "\n"
    
    prempt += prompt

    prempt += "\n\n"

    prempt += "Please determine if all the criteria have been covered, and whether any are missing. Then provide a detailed cost estimation and analysis given the parameters and the transcript provided."

    prempt += "\n\n"


    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "You are a chatbot trained to be an expert project manager and estimator"},
                {"role": "user", "content": prempt},
            ]
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    print(result)
    
    print ("******************")
    
    ##process
    
    rjson = {}
    
    
    rdata = []
    
    
    rjson['data'] = rdata
        
    
    return result, rjson







def dummy(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    if request.method == 'OPTIONS':
        # Allows GET requests from origin https://mydomain.com with
        # Authorization header
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Max-Age': '3600',
            'Access-Control-Allow-Credentials': 'true'
        }
        return ('', 204, headers)

    # Set CORS headers for main requests
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Credentials': 'true'
    }

    mongostr = os.environ.get('MONGOSTR')
    client = pymongo.MongoClient(mongostr)
    db = client["hackathon"]

    request_json = request.get_json()

    retjson = {}

    action = request_json['action']



    if action == "addprojectquestion" :
        text = request_json['text']

        col = db.projectquestions
        id = 0
        for x in col.find():
            id +=1
            if x['text'] == text:
                    
                retjson['status'] = "question already exists - not added"
                retjson['id'] = id

                return json.dumps(retjson)

        reading = {}
        reading['text'] = text

        result=col.insert_one(reading)


        retjson['status'] = "successfully added"
        retjson['id'] = id

        return json.dumps(retjson)



    if action == "addprojectvariable" :
        name = request_json['name']
        units = request_json['units']

        col = db.projectvariables
        id = 0
        for x in col.find():
            id +=1
            if x['name'] == name:
                    
                retjson['status'] = "variable already exists - not added"
                retjson['id'] = id

                return json.dumps(retjson)

        reading = {}
        reading['name'] = name
        reading['units'] = units
        

        result=col.insert_one(reading)


        retjson['status'] = "successfully added"
        retjson['id'] = id

        return json.dumps(retjson)




    if action == "addchatgptcache" :
        prompt = request_json['prompt']
        resp = request_json['response']

        col = db.chatgpt
        id = 0
        for x in col.find():
            id +=1
            if x['prompt'] == prompt:
                    
                retjson['status'] = "prompt already exists - not added"
                retjson['id'] = id

                return json.dumps(retjson)

        reading = {}
        reading['prompt'] = prompt
        reading['response'] = resp
        reading['scope'] = "projectmanager"

        result=col.insert_one(reading)


        retjson['status'] = "successfully added"
        retjson['id'] = id

        return json.dumps(retjson)



    if action == "addcosts" :
        costs = request_json['costbasis']
        id = 0
        for r in costs:
            name = r['name']
            cost = r['cost']
            qty = r['quantity']
            stock = r['stock']
            ts = r['time']

            col = db.costbasis
            
            for x in col.find():
                id +=1
          
                if x['name'] == name:

                    # col.update_one({"id": id}, {"$set":{"count":count+1}})
                        
                    retjson['status'] = "cost basis exists"
                    retjson['id'] = id

                    return json.dumps(retjson)

            reading = {}
            reading['name'] = name
            reading['cost'] = cost
            reading['qty'] = qty
            reading['stock'] = stock
            reading['time'] = ts

            result=col.insert_one(reading)

        # pid = add_readings(conn, name, ownerid, value, time)

        retjson['status'] = "successfully added"
        retjson['id'] = id

        return json.dumps(retjson)





    if action == "gptcuratedproject":
        prompt = request_json['prompt']
        token = request_json['token']


        col = db.chatgpt
        id = 0
        for x in col.find():
            id +=1
            if x['prompt'] == prompt and x['scope'] == "projectmanager2":
                    
                retjson['status'] = "prompt retrieved from cache"
                retjson['result'] = x['response']
                retjson['id'] = id

                return json.dumps(retjson)

        col2 = db.projectquestions

        result, rjson = getprojectplan1(prompt, col2)
        
        reading = {}
        reading['prompt'] = prompt
        reading['response'] = result
        reading['scope'] = "projectmanager2"

        reslt=col.insert_one(reading)


        retjson['status'] = "prompt retrieved from OpenAI"
        retjson['result'] = result

        return json.dumps(retjson)




    if action == "gptprojectmanager":
        prompt = request_json['prompt']
        token = request_json['token']


        col = db.chatgpt
        id = 0
        for x in col.find():
            id +=1
            if x['prompt'] == prompt and x['scope'] == "projectmanager3":
                    
                retjson['status'] = "prompt retrieved from cache"
                retjson['result'] = x['response']
                retjson['id'] = id

                return json.dumps(retjson)

        col2 = db.projectquestions
        col3 = db.projectvariables

        result, rjson = getprojectplan2(prompt, col2, col3)
        
        reading = {}
        reading['prompt'] = prompt
        reading['response'] = result
        reading['scope'] = "projectmanager3"

        reslt=col.insert_one(reading)


        retjson['status'] = "prompt retrieved from OpenAI"
        retjson['result'] = result

        return json.dumps(retjson)




    if action == "gptcostestimator":
        prompt = request_json['prompt']
        token = request_json['token']


        col = db.chatgpt
        id = 0
        for x in col.find():
            id +=1
            if x['prompt'] == prompt and x['scope'] == "projectmanager4":
                    
                retjson['status'] = "prompt retrieved from cache"
                retjson['result'] = x['response']
                retjson['id'] = id

                return json.dumps(retjson)

        # col2 = db.projectquestions
        col2 = db.projectvariables

        result, rjson = getprojectplan3(prompt, col2)
        
        reading = {}
        reading['prompt'] = prompt
        reading['response'] = result
        reading['scope'] = "projectmanager4"

        reslt=col.insert_one(reading)


        retjson['status'] = "prompt retrieved from OpenAI"
        retjson['result'] = result

        return json.dumps(retjson)





    if action == "gptrawproject":
        prompt = request_json['prompt']
        token = request_json['token']


        col = db.chatgpt
        id = 0
        for x in col.find():
            id +=1
            if x['prompt'] == prompt and x['scope'] == "projectmanager1":
                    
                retjson['status'] = "prompt retrieved from cache"
                retjson['result'] = x['response']
                retjson['id'] = id

                return json.dumps(retjson)



        result, rjson = getcuratedprojectplan(prompt)
        
        reading = {}
        reading['prompt'] = prompt
        reading['response'] = result
        reading['scope'] = "projectmanager1"

        reslt=col.insert_one(reading)


        retjson['status'] = "prompt retrieved from OpenAI"
        retjson['result'] = result

        return json.dumps(retjson)




    if action == 'donothing':


        retjson['status'] = "active"
        retjson['result'] = "i did nothing"

        

        return json.dumps(retjson)





    retstr = "action not done"

    if request.args and 'message' in request.args:
        return request.args.get('message')
    elif request_json and 'message' in request_json:
        return request_json['message']
    else:
        return retstr
