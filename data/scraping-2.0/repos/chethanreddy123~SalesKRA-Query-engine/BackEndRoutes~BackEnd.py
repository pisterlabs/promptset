from fastapi import FastAPI, HTTPException, Request, Response
import google.generativeai as palm
from fastapi.middleware.cors import CORSMiddleware
import pymongo
from pymongo.mongo_client import MongoClient
from loguru import logger
import random

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from langchain.callbacks import get_openai_callback
from langchain.llms import GooglePalm
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
from pymongo.mongo_client import MongoClient
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


llm = GooglePalm(
    model='models/text-bison-001',
    temperature=0,
    # The maximum length of the response
    max_output_tokens=80000,
    google_api_key='AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM'
)

MongoDB_Key = "mongodb://aioverflow:12345@ac-pu6wews-shard-00-00.me4dkct.mongodb.net:27017,ac-pu6wews-shard-00-01.me4dkct.mongodb.net:27017,ac-pu6wews-shard-00-02.me4dkct.mongodb.net:27017/?ssl=true&replicaSet=atlas-jcoztp-shard-0&authSource=admin&retryWrites=true&w=majority"
Data = MongoClient(MongoDB_Key)
EmployeeData = Data['FinalAxisBankHackathon']['EmployeeData']
KRAsData = Data['FinalAxisBankHackathon']['KRAsData']

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

def load_json_file(file_path):
    with open(file_path, 'r') as json_file:
        json_string = json_file.read()
    return json_string

documents = EmployeeData.find()

# Convert documents to JSON strings
json_string = [json.dumps(doc, default=str) for doc in documents]



# Initialize your LLM chain and conversation chain
llm_chat = GooglePalm(
    model='models/text-bison-001',
    temperature=0,
    max_output_tokens=80000,
    google_api_key='AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM'
)

conversation_buf = ConversationChain(
    llm=llm_chat,
    memory=ConversationBufferMemory()
)



# Initialize conversation_buf with the initial message
initial_message = f'''Good morning AI!, You are an Expert 
in Sales Key Result Areas (KRA) Setting and Performance Management. 
You are here to help me with my queries regarding Sales Key Result Areas 
(KRA) Setting and Performance Management, And all the sales employee data is 
given below for future analysis: {json_string}'''


check = conversation_buf(initial_message)

palm.configure(api_key='AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM')
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
response = palm.chat(messages=["Jon is very good boy and works and microsoft"])
MongoDB_Key = "mongodb://aioverflow:12345@ac-pu6wews-shard-00-00.me4dkct.mongodb.net:27017,ac-pu6wews-shard-00-01.me4dkct.mongodb.net:27017,ac-pu6wews-shard-00-02.me4dkct.mongodb.net:27017/?ssl=true&replicaSet=atlas-jcoztp-shard-0&authSource=admin&retryWrites=true&w=majority"
Data = MongoClient(MongoDB_Key)
EmployeeData = Data['FinalAxisBankHackathon']['EmployeeData']
KRAsData = Data['FinalAxisBankHackathon']['KRAsData']


conversation_buf_react = ConversationChain(
    llm=llm_chat,
    memory=ConversationBufferMemory()
)



# Initialize conversation_buf with the initial message
initial_message_react = f'''Act like expert Data analyst and react developer'''


check_react = conversation_buf_react(initial_message_react)


############### Additional Functions #####################

def generate_random_id():
    # Generate a random 6-digit number
    random_number = random.randint(100000, 999999)
    
    # Combine with the fixed string 'AXIS'
    new_id = f'AXIS{random_number}'
    
    return new_id

def generate_random_id_KRA():
    # Generate a random 6-digit number
    random_number = random.randint(100000, 999999)
    
    # Combine with the fixed string 'AXIS'
    new_id = f'KRA{random_number}'
    
    return new_id

############### API Endpoints #####################

# Define API endpoint for chat
@app.post("/chat/")
async def chat(request: Request):
    # json_data = await request.json()
    # query = json_data.get("query")

    # try:
    #     response = count_tokens(conversation_buf, query)
    #     return {"response": response}
    # except Exception as e:
    #     return HTTPException(status_code=500, detail=str(e))

    image_path = "Screenshot 2023-09-14 at 7.00.54 PM.png"  # Update with your image file path
    return FileResponse(image_path)


@app.post("/chat_react/")
async def chat(request: Request):
    json_data = await request.json()
    query = json_data.get("query")

    try:
        response = count_tokens(conversation_buf_react, query)
        return {"response": response}
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))


@app.post("/AddEmployee/")
def NewPatient(info : dict):
    req_info = info
    req_info = dict(req_info)
    print(req_info)
    logger.info("recieved new patient details")
    req_info['personalInformation']['EmployeeID'] = generate_random_id()
    req_info['ListOfKRAs'] = []

    try:
        Check = EmployeeData.insert_one(req_info)
        if Check.acknowledged == True:
            logger.info("patient added successfully")
            return {"status": "success" , "EmployeeID": req_info['personalInformation']['EmployeeID']}
        else:
            logger.info("patient not added")
            return {"status": "failed"}
    except Exception as e:
        logger.error(e)
        return {"status": "failed"}
    
@app.post("/GetEmployee/")
def GetEmployee(info : dict):
    req_info = info
    req_info = dict(req_info)
    EmployeeID = req_info['EmployeeID']
    Result = EmployeeData.find_one({"personalInformation.EmployeeID": EmployeeID})
    if Result is None:
        return {"status": "failed"}
    else:
        del Result['_id']
        return Result


@app.post("/AddKRAtoEmployee/")
def AddKRAtoEmployee(info : dict):
    req_info = info
    req_info = dict(req_info)
    EmployeeID = req_info['EmployeeID']
    KRAID = req_info['KRAID']
    Result = EmployeeData.find_one({"personalInformation.EmployeeID": EmployeeID})
    if Result is None:
        return {"status": "failed"}
    else:
        del Result['_id']
        Result['ListOfKRAs'].append(KRAID)
        Check = EmployeeData.update_one({"personalInformation.EmployeeID": EmployeeID}, {"$set": Result})
        if Check.acknowledged == True:
            return {"status": "success"}
        else:
            return {"status": "failed"}


@app.get("/GetAllEmployees/")
def GetAllEmployees():
    logger.info("recieved all employee details")
    Result = list(EmployeeData.find({}))
    if Result is None:
        return {"status": "failed"}
    else:
        for i in Result:
            del i['_id']
        return Result


@app.post("/AddKRA/")
def AddKRA(info : dict):
    req_info = info
    req_info = dict(req_info)
    print(req_info)
    logger.info("recieved new patient details")
    req_info['KRAID'] = generate_random_id_KRA()
    try:
        Check = KRAsData.insert_one(req_info)
        if Check.acknowledged == True:
            logger.info("patient added successfully")
            return {"status": "success" , "KRAID": req_info['KRAID']}
        else:
            logger.info("patient not added")
            return {"status": "failed"}
    except Exception as e:
        logger.error(e)
        return {"status": "failed"}
    
@app.get("/GetAllKRAs/")
def GetAllKRAs():
    logger.info("recieved all employee details")
    Result = list(KRAsData.find({}, {'KRAID': 1, '_id': 0, 'title' : 1}))
    if Result is None:
        return {"status": "failed"}
    else:
        for i in Result:
            i['value'] = i['KRAID'] + " " + i['title']
            i["label"] = i['KRAID'] + " " + i['title']
            del i['KRAID']
            del i['title']
        return Result
    

@app.get("/GetAllKRAsData/")
def GetAllKRAsData():
    logger.info("recieved all employee details")
    Result = list(KRAsData.find({}))
    if Result is None:
        return {"status": "failed"}
    else:
        for i in Result:
            del i['_id']
        return Result


@app.post("/GetKRA/")
def GetKRA(info : dict):

    req_info = info
    req_info = dict(req_info)
    KRA_ID = req_info['KRAID']
    Result = KRAsData.find_one({"KRAID": KRA_ID})
    if Result is None:
        return {"status": "failed"}
    else:
        del Result['_id']
        return Result
    


@app.post("/GetKRAsForEmployee/")
def GetKRAsForEmployee(info : dict):

    req_info = info
    req_info = dict(req_info)
    EmployeeID = req_info['EmployeeID']
    Result = EmployeeData.find_one({"personalInformation.EmployeeID": EmployeeID})
    if Result is None:
        return {"status": "failed"}
    else:
        del Result['_id']
        return Result['ListOfKRAs']