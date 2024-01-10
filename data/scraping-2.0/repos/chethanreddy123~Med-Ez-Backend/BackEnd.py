 
# Importing all the necessary libraries

import re
import math
import pymongo
from fastapi import FastAPI, File, UploadFile
from fastapi import FastAPI, HTTPException
from collections import Counter
from urllib import request
import pandas as pd
from langchain.llms import GooglePalm
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as palm
import json
import easyocr
import subprocess
from reportgenerator import create_pdf_discharge
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder
from pymongo.mongo_client import MongoClient
import joblib
from fastapi import Response, BackgroundTasks
import random as rd
import datetime
from loguru import logger
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import StreamingResponse
from pymongo import MongoClient
from typing import List, Optional
import json
from datetime import date
from bson.timestamp import Timestamp
import datetime as dt


# mongoDB connection keys for both local and cloud


Key_Mongo_Cloud = "mongodb://aioverflow:12345@ac-pu6wews-shard-00-00.me4dkct.mongodb.net:27017,ac-pu6wews-shard-00-01.me4dkct.mongodb.net:27017,ac-pu6wews-shard-00-02.me4dkct.mongodb.net:27017/?ssl=true&replicaSet=atlas-jcoztp-shard-0&authSource=admin&retryWrites=true&w=majority"
Key_Mongo_Local = "mongodb://localhost:27017/"

# Connecting to the database

Data = MongoClient(Key_Mongo_Cloud)
PatientData = Data['HoepHackathon']['PatientData']
LoginDatabase = Data['HoepHackathon']['LoginCred']
ReviewData = Data['HoepHackathon']['Reviews']
ReHab = Data['HoepHackathon']['Re-Hab']
ReVisit = Data['HoepHackathon']['ReVisitPopUps']
SearchIndex = Data['HoepHackathon']['Patient_Search']
billData = Data['HoepHackathon']['Bills']
currId = Data['HoepHackathon']['PatientID']


# creating fastapi instance

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### ----- General Functions ---------- #####3


with open('restructured_exercise.json', 'r') as json_file:
    exercise_links = json.load(json_file)

def Dict_to_List(Dictionary):
    return [i['value'] for i in Dictionary]

def process_dictionary(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list) and all(isinstance(item, dict) and "value" in item and "label" in item for item in value):
                data[key] = [item["value"] for item in value]
            else:
                process_dictionary(value)
    elif isinstance(data, list):
        for i in range(len(data)):
            process_dictionary(data[i])

def convert_to_second_json_format(first_json):
    second_json = {
        "Basic": {
            "Patient_Id": first_json["Patient_Id"],
            "DateOfAssessment": first_json["DateOfAssessment"],
            "Patient_Name": first_json["Patient_Name"],
            "Patient_Age": first_json["Patient_Age"],
            "Patient_Weight": first_json["Patient_Weight"],
            "Patient_Height": first_json["Patient_Height"],
            "Patient_Contact_No": first_json["Patient_Contact_No"],
            "Diagnosis": first_json["Diagnosis"],
            "TreatmentGiven": first_json["TreatmentGiven"],
            "Package": first_json["Package"],
            "FollowUp": first_json["FollowUp"],
            "ReviewDate": first_json["ReviewDate"],
            "Contradiction": first_json["Contradiction"],
            "Category": first_json["Category"],
            "InvestigationDone": first_json["InvestigationDone"],
            "TargetingMuscle": first_json["TargetingMuscle"],
            "TargetingJoint": first_json["TargetingJoint"],
            "PainScale": first_json["PainScale"],
            "AssessmentDoneBy": first_json["AssessmentDoneBy"]
        },
        "ExerciseSchedule": first_json["ExerciseSchedule"],
        "ExerciseTracking": first_json["ExerciseTracking"],
        "PARQ_Assessment": first_json["PARQ_Assessment"],
        "ScheduleDoneBy": first_json["ScheduleDoneBy"],
        "TrainerName": first_json["TrainerName"]
    }
    return second_json

# Utility function to remove _id field from documents
def remove_id(documents: List[dict]) -> List[dict]:
    return [{**doc, "_id": None} for doc in documents]

# Background job to process the data and generate the JSON response
def process_patients(background_response: BackgroundTasks):
    Find = PatientData.find({})
    Find = list(Find) if Find is not None else []
    if len(list(Find)) == 0:
        return None

    Result = list(Find)
    Result = remove_id(Result)

    for i in Result:
        LastAsses = i.get('Assessment', [])
        Checker = len(LastAsses)
        if Checker == 0:
            i['LastAssessment'] = 'No Assessment'
            i['Status'] = "Not Yet"
        else:
            i['LastAssessment'] = LastAsses[-1]
            SeniorDoctorPrescription = i['LastAssessment'].get('SeniorDoctorPrescription', {})
            TreatmentPrescription = SeniorDoctorPrescription.get('TreatmentPrescription', {})
            if TreatmentPrescription != {}:
                i['Status'] = "Completed"
            else:
                i['Status'] = "Partial"

    return Result

# Helper function to generate streaming JSON response
def stream_generator(background_response: BackgroundTasks):
    patients = process_patients(background_response)
    if patients is None:
        yield "[]"
        return

    yield "["
    first = True
    for patient in patients:
        if not first:
            yield ","
        yield json.dumps(patient)
        first = False
    yield "]"

def check_dict_fields(dictionary):
    print("Hello")
    del dictionary['Occupation']
    for value in dictionary.values():
        if value is None or value == "":
            return False
    return True

### ids generator ###


def get_latest_id():
    Find = currId.find({})
    Find = list(Find) 
    Find = Find[0]
    return Find['currId']

def update_latest_id(new_id):
    currId.update_one({}, {"$set": {"currId": new_id}})

def create_id():
    latest_id = get_latest_id()

    # Extract the numeric part of the latest ID
    numeric_part = int(latest_id[5:])

    # Generate the new ID by incrementing the numeric part
    new_numeric_part = numeric_part + 1
    new_id = f'23MED{new_numeric_part:04d}'

    # Update the latest ID in the text file
    update_latest_id(new_id)

    return new_id

### OpenText and LLM Routes #####

reader = easyocr.Reader(['en'])

# Initialize GooglePalm
Palm = GooglePalm(temperature=0, 
                 model="models/text-bison-001", 
                 google_api_key="AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM")

@app.post("/extract_info/")
async def extract_info(image: UploadFile = File(...)):
    logger.info("Get Extract info")

    file_extension =  image.filename.split(".")[-1]
    
    # Save the uploaded file
    file_path = f"uploads/{image.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(image.file.read())
    
    if file_extension.lower() in ['png', 'jpg', 'jpeg']:
        result = reader.readtext(file_path)
        labels = [bbox_label[1] for bbox_label in result]
        raw_string = ' '.join(labels)
    else:
        return {"error": "Unsupported file format"}
    extracted_info = []
    # Run LLMChain:
    # Initialize LLMChain
    template = '''Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'information_extraction' function.

    Passage:
    {raw_text}

    schema = {{
        "properties" : {{
            "name" : {{"type" : "string"}},
            "email" : {{"type" : "string"}},
            "age" : {{"type" : "string"}},
            "height" : {{"type" : "integer"}},
            "weight" : {{"type" : "integer"}},
            "phone" : {{"type" : "integer"}},
            "gender" : {{"type" : "string"}},
            "address" : {{"type" : "string"}}
        }},
        "required" : ["name" , "email" , "age" , "height" , "weight" , "phone" , "gender" , "address"]
    }}

    Note: If values or not extracted Make them ''.
    '''

    prompt = PromptTemplate(template=template, input_variables=["raw_text"])

    llm_chain = LLMChain(prompt=prompt, llm=Palm)
    res = llm_chain.run(raw_string)
    info = json.loads(res)
    extracted_info.append(info)
    
    return {"extracted_info": extracted_info}

@app.post("/generate_prescription/")
def generate_prescription(info : dict):
    req_info = info
    req_info = dict(req_info)
    patient_id = req_info['patient_id']
    date_of_assessment = req_info['DateOfAssessment']
    logger.info("Got the request for generating prescription")
    result = PatientData.find_one({"Patient_Id" : patient_id , "Assessment.Date" : date_of_assessment})

    if result is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    current_ass = None
    for i in result['Assessment']:
        if i['Date'] == date_of_assessment:
            current_ass = i
            break

    template = '''Act like an expert medical analyst in the domain of physiotherapy.
    Below is a JSON file containing the assessment data of a specific patient.
    Write a short summary (200 words) for the same:

    Patient General Assessment in JSON format:
    {gen_assessment}
    '''
    Palm = GooglePalm(temperature=0, 
                     model="models/text-bison-001" , 
                     google_api_key="AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM")


    prompt = PromptTemplate(template=template, input_variables=["gen_assessment"])

    llm_chain = LLMChain(
        prompt=prompt,
        llm=Palm
    )

    genSum = llm_chain.run(str(current_ass))


    template = '''Act like an expert medical analyst in the domain of physiotherapy.
        Below is summary of general assessment of a specific patient.
        Write prescription (list attributes given below) for this patient.

        [
            "DateOfAssessment",
            "diagnosis",
            "reviewNext",
            "treatmentPlan",
            "numberOfDays",
            "contraindication",
            "followUp",
            "homeAdvice",
            "exercises"
        ]


        Patient General Assessment:
        {gen_assessment}

    '''
    Palm = GooglePalm(temperature=0, 
                         model="models/text-bison-001" , 
                         google_api_key="AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM")


    prompt = PromptTemplate(template=template, input_variables=["gen_assessment"])

    llm_chain = LLMChain(
        prompt=prompt,
        llm=Palm
    )

    res = llm_chain.run(str(genSum).replace("*" , ""))
    
    return res

@app.post("/generate_summary/")
def generate_summary(info : dict):

    req_info = info
    req_info = dict(req_info)
    patient_id = req_info['patient_id']
    date_of_assessment = req_info['DateOfAssessment']
    logger.info("Got the request for generating summary")
    result = PatientData.find_one({"Patient_Id" : patient_id , "Assessment.Date" : date_of_assessment})
    if result is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    NewDict = {}
    patient_info_keys = [
        'Patient_Id',
        'Patient_Name',
        'Patient_Age',
        'Patient_Gender',
        'Patient_Height',
        'Patient_Weight',
        'Patient_Contact_No'
    ]

    current_ass = None
    for i in result['Assessment']:
        if i['Date'] == date_of_assessment:
            current_ass = i
            break
    for i in patient_info_keys:
        NewDict[i] = result[i]
    for key, value in current_ass.items():
        if type(value) != dict:
            NewDict[key] = value
    

    template = '''Act like an expert medical analyst in the domain of physiotherapy and content analyser.
    Below is a JSON file containing the assessment data of a specific patient. 
    Write a summary of the patient's condition (in paragraph) in approximately 100 words 
    for the senior doctor to continue with the treatment.

    Patient Data in JSON format:
    {patient_data}

    Output: Summary of about 100 words
    '''

    logger.info(NewDict)
    
    Palm = GooglePalm(temperature=0, model="models/text-bison-001" , google_api_key="AIzaSyA1fu-ob27CzsJozdr6pHd96t5ziaD87wM")
    prompt = PromptTemplate(template=template, input_variables=["patient_data"])
    llm_chain = LLMChain(prompt=prompt, llm=Palm)
    res = llm_chain.run(str(NewDict))
    logger.info("Summary generated successfully")

    return {"summary": res}


############### --------------------------- Login Routes ---------------------------- ##############

@app.post("/loginCheck")
def loginCheck(info : dict):
    
    req_info = info
    req_info = dict(req_info)
    logger.info("recieved login info")

    Result = (LoginDatabase.find_one(req_info))


    if Result != None:
        Result = dict(Result)
        if req_info['userName'] == Result['userName'] and req_info['password'] == Result['password'] and req_info['userType'] == Result['userType']:
            return {
                "Status" : True
            }
        else:
            return {
                "Status" : False
            }
        
    else:
        return {
            "Status" : False
        }

#################  -------------------- Receptionist Routes ------- ###################### 

@app.post("/newPatient")
def NewPatient(info : dict):

    req_info = info
    req_info = dict(req_info)
    print(req_info)
    logger.info("recieved new patient details")

    Dictionary = req_info.copy()
    Checker = check_dict_fields(Dictionary)

    if Checker == False:
        return {"Status" : "Fields are empty"}
    
    new_patient_id = create_id()

    CurrentData = {
                "Patient_Id" : new_patient_id,
                "Patient_Name" : req_info["Patient_Name"],
                "Patient_Age" : req_info["Patient_Age"],
                "Patient_Gender" : req_info["Patient_Gender"],
                "Patient_Height" : req_info["Patient_Height"],
                "Patient_Weight" : req_info["Patient_Weight"],
                "Patient_Contact_No" : req_info["Patient_Contact_No"],
                "Patient_Email" : req_info["Patient_Email"],
                "Employed" : req_info["Employed"],
                "Occupation" : req_info["Occupation"],
                "Address" : req_info["Address"],
                "Assessment" : [],
                "createdAt" : dt.datetime.today().timestamp()
            }
    SearchData = {
        "Patient_Id" : CurrentData['Patient_Id'],
        "Patient_Name" : req_info['Patient_Name'],
        "Patient_Gender" : req_info['Patient_Gender'],
        "Patient_Age" : req_info['Patient_Age'],
        "Patient_Contact_No" : req_info['Patient_Contact_No']
    }
    

    ReturnObj = dict(CurrentData)    
    Check1 = PatientData.insert_one(CurrentData)
    Check2 = SearchIndex.insert_one(SearchData)
    if Check1.acknowledged == True and Check2.acknowledged == True:
        return ReturnObj
    else:
        return {"Status" :  False}
    
@app.post("/addBasicAssessment")
def addBasicAssessment(info : dict):

    # 
    req_info = info
    req_info = dict(req_info)
    logger.info("recieved basic assesment")

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})


    if Find == None:
        return {"Status" : "Patient Not Found"}
    else:
        Result = dict(Find)
        # print(Result)


        # --- MakesConditionWorse Pre-Processing ------- #
        
        MakesConditionWorse = [i['value'] for i in req_info['Assessment']['MakesConditionWorse']]
        # print(MakesConditionWorse)
        req_info['Assessment']['MakesConditionWorse'] = MakesConditionWorse

        MakesConditionBetter = [i['value'] for i in req_info['Assessment']['MakesConditionBetter']]
        # print(MakesConditionBetter)
        req_info['Assessment']['MakesConditionBetter'] = MakesConditionBetter

        MedicalInformation = [i['value'] for i in req_info['Assessment']['MedicalInformation']]
        # print(MedicalInformation)
        req_info['Assessment']['MedicalInformation'] = MedicalInformation

        MedicalIntervention = [i['value'] for i in req_info['Assessment']['MedicalIntervention']]
        # print(MedicalIntervention)
        req_info['Assessment']['MedicalIntervention'] = MedicalIntervention

        UpdateDict = req_info['Assessment']
        UpdateDict["SeniorDoctorPrescription"] = dict()
        UpdateDict["JuniorDoctorPrescription"] = dict({"DayWise" : []})
        UpdateDict["TrainerPrescription"] = dict()
        UpdateDict["Feedback"] = {}
        UpdateDict["SeniorWrittenPres"] = False
        UpdateDict["SeniorWrittenAsses"] = False

        # print(UpdateDict)

        Result['Assessment'].append(UpdateDict)
        UpdateAssigment = Result['Assessment']

        # print(UpdateAssigment)

        myquery = { "Patient_Id": SearchKey }
        newvalues = { "$set": { "Assessment": UpdateAssigment } }
        PatientData.update_one(myquery, newvalues)
    
        return {"Status" : "Successfully"}
    
@app.post("/SearchPatient")
def SearchPatient(info : dict):

    # 
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['SearchString']

    if SearchKey == "":
        pipeline = [
            {
                "$project": {
                    "_id": 0,
                    "Patient_Id": 1,
                    "Patient_Name": 1,
                    "Patient_Age": 1,
                    "Patient_Gender": 1,
                    "Patient_Contact_No": 1,
                    "LastAssessment": {
                        "$cond": {
                            "if": { "$eq": [ { "$size": { "$ifNull": ["$Assessment", []] } }, 0 ] },
                            "then": {},
                            "else": { "$arrayElemAt": [ "$Assessment", -1 ] }
                        }
                    },
                    "createdAt": 1  # Assuming "createdAt" field represents the insertion timestamp
                }
            },
            {
                "$project": {
                    "Patient_Id": 1,
                    "Patient_Name": 1,
                    "Patient_Age": 1,
                    "Patient_Gender": 1,
                    "Patient_Contact_No": 1,
                    "LastAssessment.Date": 1,
                    "LastAssessment.Complaint": 1,
                    "createdAt": 1
                }
            },
            { "$sort": { "createdAt": -1 } },  # Sort by createdAt field in descending order
            { "$limit": 10 }
        ]

        result = list(PatientData.aggregate(pipeline))

        return {"allPatients": result}


    results = SearchIndex.aggregate([
        {
            "$search": {
                "index": "Patient_Search",
                "text": {
                    "query": SearchKey,
                    "path": ["Patient_Id","Patient_Contact_No","Patient_Name"],
                    "fuzzy": {}
                }
            }
        },
        {
            "$limit" : 5
        }
    ])

    FinalResutls = list(results)

    for i in FinalResutls:
        del i['_id']


    return {"Results" : FinalResutls}

@app.post("/viewPatient")
def viewPatient(info : dict):

    # 
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found"}
    else:
        Result = dict(Find)
        del Result['_id']
        # process_dictionary(Result)
        return Result

@app.get("/allPatientsOld") # only Top 10 patients will be shown
def allPatientsOld():
    # Find = PatientData.find().limit(5).sort([("$natural", -1)])
    Find = PatientData.find({})
    if Find == None:
        return {"Status" : "Patient Not Found"}
    else:
        Result = list(Find)
        for i in Result:
            del i['_id']
        for i in Result:
            LastAsses = i['Assessment']
            Checker = len(LastAsses)
            if Checker == 0:
                i['LastAssessment'] = 'No Assessment'
                i['Status'] = "Not Yet"
            else:
                i['LastAssessment'] = LastAsses[len(LastAsses) - 1]
                if "TreatmentPrescription" in i['LastAssessment']['SeniorDoctorPrescription']:
                    if i['LastAssessment']['SeniorDoctorPrescription']['TreatmentPrescription'] != dict():
                        i['Status'] = "Completed"
                    else:
                        i['Status'] = "Partial"
                else:
                    i['Status'] = "Not Yet"
        
        
        return {"allPatients" : Result}

@app.get("/allPatients")
def allPatients():
    pipeline = [
        {
            "$project": {
                "_id": 0,
                "Patient_Id": 1,
                "Patient_Name": 1,
                "Patient_Age": 1,
                "Patient_Gender": 1,
                "Patient_Contact_No": 1,
                "LastAssessment": {
                    "$cond": {
                        "if": { "$eq": [ { "$size": { "$ifNull": ["$Assessment", []] } }, 0 ] },
                        "then": {},
                        "else": { "$arrayElemAt": [ "$Assessment", -1 ] }
                    }
                },
                "createdAt": 1  # Assuming "createdAt" field represents the insertion timestamp
            }
        },
        {
            "$project": {
                "Patient_Id": 1,
                "Patient_Name": 1,
                "Patient_Age": 1,
                "Patient_Gender": 1,
                "Patient_Contact_No": 1,
                "LastAssessment.Date": 1,
                "LastAssessment.Complaint": 1,
                "createdAt": 1
            }
        },
        { "$sort": { "createdAt": -1 } },  # Sort by createdAt field in descending order
        { "$limit": 10 }
    ]

    result = list(PatientData.aggregate(pipeline))
    return {"allPatients": result[::-1]}

# Endpoint for retrieving paginated patients
@app.get("/allPatientsFaster/", response_model=List[dict])
def all_patients(
    skip: int = 0,
    limit: int = 10
):
    background_response = BackgroundTasks()
    background_response.add_task(process_patients, background_response)

    return StreamingResponse(
        stream_generator(background_response),
        media_type="application/json"
    )

# Additional endpoints can be added here for filtering or specific patient retrieval
@app.post("/updatePatient")
def updatePatient(info : dict):

    # 
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found" }
    else:
        Result = dict(Find)
        check = PatientData.update_one({'Patient_Id' : req_info["Patient_Id"]},
                   {"$set": {
                       "Patient_Name": req_info["Patient_Name"],
                       "Patient_Age":  req_info["Patient_Age"],
                       "Patient_Gender" : req_info["Patient_Gender"],
                       "Patient_Height" : req_info["Patient_Height"],
                       "Patient_Weight" : req_info["Patient_Weight"],
                       "Patient_Contact_No" : req_info["Patient_Contact_No"],
                       "Employed" : req_info["Employed"],
                       "Occupation" : req_info["Occupation"],
                       "Address" : req_info["Address"]
                        }
                   }
                )
        
        if check.acknowledged == True:
            return {"Status" : True ,  "Patient_Id" : req_info["Patient_Id"]}
        else:
            return {"Status" : False , "Patient_Id" : req_info["Patient_Id"]}
        
@app.post("/GetDischargeSummary")
def GetDischargeSummary(info : dict):

    # 
    req_info = info
    req_info = dict(req_info)
    print(req_info)

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found" }
    name = Find['Patient_Name']
    age = Find['Patient_Age']
    gender = Find['Patient_Gender']
    currAssessment = None
 

    currAssessment = Find['Assessment'][-1]

    referred_by = currAssessment['ReferalDoctor']
    chief_complaint = currAssessment['Complaint']
    previous_treatment = currAssessment['RecievedTherapy']
    diagnosis = currAssessment['SeniorDoctorPrescription']['TreatmentPrescription']['diagnosis']
    duration = currAssessment['SeniorDoctorPrescription']['TreatmentPrescription']['numberOfDays']
    treatment_given = currAssessment['SeniorDoctorPrescription']['TreatmentPrescription']['treatmentPlan']

    # treatment_dates = ["2023-07-25", "2023-07-26", "2023-07-27", "2023-07-28", "2023-07-29"] # from junior
    # painscales = [2, 3, 1, 2, 1] # from junior

    treatment_dates = [i['Date'] for i in currAssessment['JuniorDoctorPrescription']['DayWise']]
    painscales = [i['PainScale'] for i in currAssessment['JuniorDoctorPrescription']['DayWise']]
    home_advice = currAssessment['SeniorDoctorPrescription']['TreatmentPrescription']['homeAdvice']
    next_review = currAssessment['SeniorDoctorPrescription']['TreatmentPrescription']['reviewNext']
    doctor_prescription = currAssessment['JuniorDoctorPrescription']['DayWise']
    exercises = currAssessment['SeniorDoctorPrescription']['TreatmentPrescription']['exercises']

    create_pdf_discharge(name, age, gender, referred_by, chief_complaint, previous_treatment, diagnosis, duration,
            treatment_given, treatment_dates, painscales, home_advice, next_review , doctor_prescription,
            exercises)

    return FileResponse("hospital_report.pdf")
    
@app.post("/GetRehabBill")
def GetRehabBill(info : dict):

    # 
    req_info = info
    req_info = dict(req_info)

    bill_no = rd.randint(1,100)
    patient_id = req_info['Patient_Id']
    date = req_info['date']
    name = req_info['name']
    address = req_info['address']
    cell_no = req_info['cell_no']
    amount_paid = req_info['amount_paid']
    package_program = req_info['package_program']

    subprocess.run(["python", "generaterehabbill.py", str(bill_no), str(patient_id), str(date), str(name), str(address), str(cell_no), str(amount_paid), str(package_program)])


    # create_billing_slip_rehab(bill_no, patient_id, date, name, address, cell_no, amount_paid, package_program)
    billData.insert_one(req_info)

    return FileResponse("billing_slip_rehab.pdf")

@app.post("/GetNormalBill")
def GetNormalBill(info : dict):

    # 
    req_info = info
    req_info = dict(req_info)

    bill_no = str(rd.randint(1,100))
    patient_id = req_info['Patient_Id']
    date = req_info['date']
    name = req_info['name']
    address = req_info['address']
    cell_no = req_info['cell_no']
    amount_paid = req_info['amount_paid']
    no_days = req_info['no_days']

    print(bill_no, patient_id, date, name, address, cell_no, amount_paid, no_days)

    subprocess.run(["python", "generatenormalbill.py", str(bill_no), str(patient_id), str(date), str(name), str(address), str(cell_no), str(amount_paid), str(no_days)])

    billData.insert_one(req_info)

    return FileResponse("billing_slip.pdf")

@app.post("/patientFeedback")
def patientFeedback(info : dict):

    # 
    req_info = info
    req_info = dict(req_info)

    # print(req_info)

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found"}
    else:
        Result = dict(Find)
        AllAsses = Result['Assessment']
        UpdateFeed = None

        idx = 0
        for i in AllAsses:
            if i['Date'] == req_info['Date'] :
                UpdateFeed = i
                break
            idx += 1
        if UpdateFeed == None:
            return {"Status" : "Date of Assessment doesn't exits"}
        else:
            UpdateFeed['Feedback'] = req_info['Feedback']
            AllAsses[idx] = UpdateFeed
            check = PatientData.update_one({'Patient_Id' : req_info["Patient_Id"]},
                   {"$set": {
                       "Assessment": AllAsses,
                        }
                   }
                )
            
            if check.acknowledged == True:
                return {"Status" : True}
            else:
                return {"Status" : False}



################ -------------------------- Senior / Junior Doctor Routes --------------------------------- ################

@app.get("/allPatientsTodayCount")
def allPatientsTodayCount():
    Find = PatientData.find({})
    Results = list(Find)

    DatedPatients = []

    for Data in Results:
        for i in Data['Assessment']:
            if i['Date'] == str(datetime.date.today()) and i['SeniorWrittenPres'] == False:
                # print("Happy")
                del Data['_id']
                Data['LastAssessment'] = i
                DatedPatients.append(Data)
                # print(Data)
                break
    

    return {
        "allPatientsTodayCount" : len(DatedPatients)
    }



# @app.get("/allPatientsToday")
# def allPatientsToday():

    query = {
        "Assessment": {
            "$elemMatch": {
                "Date": str(datetime.date.today()),
            }
        }
    }
    print(str(datetime.date.today()))
    Find = PatientData.find(query)
    
    Results = list(Find)
    print(Results)

    DatedPatients = []



    for Data in Results:
        for i in Data['Assessment']:
            if i['Date'] == str(datetime.date.today()) and i['SeniorWrittenPres'] == False:
                # print("Happy")
                del Data['_id']
                Data['LastAssessment'] = i
                Data['Status'] = None
                if "TreatmentPrescription" in i['SeniorDoctorPrescription']:
                    if i['SeniorDoctorPrescription']['TreatmentPrescription'] != dict():
                        Data['Status'] = "Completed"
                    else:
                        Data['Status'] = "Partial"
                else:
                    Data['Status'] = "Not Yet"
                DatedPatients.append(Data)
                break
    

    return {
        "allPatientsToday" : DatedPatients
    }


@app.get("/allPatientsToday")
def allPatientsToday():
    pipeline = [
        {
            "$unwind": "$Assessment"
        },
        {
            "$match": {
                "Assessment.Date": str(datetime.date.today()),
                "Assessment.SeniorWrittenPres": False
            }
        },
        {
            "$project": {
                "_id": 0,
                "Patient_Id": 1,
                "Patient_Name": 1,
                "Patient_Age": 1,
                "Patient_Gender": 1,
                "Patient_Contact_No": 1,
                "Status": {
                    "$cond": {
                        "if": {
                            "$and": [
                                { "$eq": [ "$Assessment.SeniorWrittenPres", False ] },
                                { "$ne": [ "$Assessment.SeniorDoctorPrescription", {} ] },
                                { "$ne": [ "$Assessment.SeniorDoctorPrescription.TreatmentPrescription", {} ] }
                            ]
                        },
                        "then": "Completed",
                        "else": {
                            "$cond": {
                                "if": {
                                    "$and": [
                                        { "$eq": [ "$Assessment.SeniorWrittenPres", False ] },
                                        { "$eq": [ "$Assessment.SeniorDoctorPrescription", {} ] }
                                    ]
                                },
                                "then": "Partial",
                                "else": "Not Yet"
                            }
                        }
                    }
                },
                "LastAssessment": {
                    "Date": "$Assessment.Date",
                    "Complaint": "$Assessment.Complaint"
                },
                "createdBy": 1
            }
        },
        {
            "$sort": {
                "createdBy": 1
            }
        }
    ]

    results = list(PatientData.aggregate(pipeline))
    return {"allPatientsToday": results}

## Adding Assessments;
@app.post("/ShoulderAssessment")
def ShoulderAssessment(info : dict):

    
    req_info = info
    req_info = dict(req_info)

    print(req_info)

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found" }
    else:
        # Update the document in MongoDB:
        del req_info['Patient_Id']

        # ListChanges = ["medKCO","personal","duration","painAss","irritability"]
        # for i in ListChanges:
        #     req_info[i] = Dict_to_List(req_info[i])
        # print(req_info)

        PatientData.update_one(
            {"Patient_Id": SearchKey, "Assessment.Date": req_info['DateOfAssessment']},
            {"$set": {
                "Assessment.$.SeniorDoctorPrescription.ShoulderAssessment": 
                    req_info
                
            }}
        )

        #medKCO,personal,duration,painAss,irritability:

        return {"Status" : "Successful"}

@app.post("/KneeAssessment")
def KneeAssessment(info : dict):

    
    req_info = info
    req_info = dict(req_info)

    print(req_info)

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found" }
    else:
        # Update the document in MongoDB:
        del req_info['Patient_Id']

        # ListChanges = ["medKCO","personal","duration","painAss","irritability"]
        # for i in ListChanges:
        #     req_info[i] = Dict_to_List(req_info[i])
        # print(req_info)

        PatientData.update_one(
            {"Patient_Id": SearchKey, "Assessment.Date": req_info['DateOfAssessment']},
            {"$set": {
                "Assessment.$.SeniorDoctorPrescription.KneeAssessment": 
                    req_info
                
            }}
        )

        #medKCO,personal,duration,painAss,irritability:

        return {"Status" : "Successful"}  

@app.post("/BalanceAssessment")
def BalanceAssessment(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    print(req_info)

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found" }
    else:
        # Update the document in MongoDB:
        del req_info['Patient_Id']

        # ListChanges = ["medKCO","personal","duration","painAss","irritability"]
        # for i in ListChanges:
        #     req_info[i] = Dict_to_List(req_info[i])
        # print(req_info)

        PatientData.update_one(
            {"Patient_Id": SearchKey, "Assessment.Date": req_info['DateOfAssessment']},
            {"$set": {
                "Assessment.$.SeniorDoctorPrescription.BalanceAssessment": 
                    req_info
                
            }}
        )

        #medKCO,personal,duration,painAss,irritability:

        return {"Status" : "Successful"}  

@app.post("/LowBackAssessment")
def LowBackAssessment(info : dict):

    # 
    req_info = info
    req_info = dict(req_info)

    print(req_info)

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found" }
    else:
        # Update the document in MongoDB:
        del req_info['Patient_Id']

        # ListChanges = ["medKCO","personal","duration","painAss","irritability"]
        # for i in ListChanges:
        #     req_info[i] = Dict_to_List(req_info[i])
        # print(req_info)

        PatientData.update_one(
            {"Patient_Id": SearchKey, "Assessment.Date": req_info['DateOfAssessment']},
            {"$set": {
                "Assessment.$.SeniorDoctorPrescription.LowBackAssessment": 
                    req_info
                
            }}
        )

        #medKCO,personal,duration,painAss,irritability:

        return {"Status" : "Successful"}

@app.post("/PARQPlusAssessment")
def PARQPlusAssessmen(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    print(req_info)

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found" }
    else:
        # Update the document in MongoDB:
        del req_info['Patient_Id']

        # ListChanges = ["medKCO","personal","duration","painAss","irritability"]
        # for i in ListChanges:
        #     req_info[i] = Dict_to_List(req_info[i])
        # print(req_info)

        PatientData.update_one(
            {"Patient_Id": SearchKey, "Assessment.Date": req_info['DateOfAssessment']},
            {"$set": {
                "Assessment.$.SeniorDoctorPrescription.PARQPlusAssessment": req_info
            }}
        )

        #medKCO,personal,duration,painAss,irritability:

        return {"Status" : "Successful"}

@app.post("/FMSAssessment")
def FMSAssessment(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    print(req_info)

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found" }
    else:
        # Update the document in MongoDB:
        del req_info['Patient_Id']

        # ListChanges = ["medKCO","personal","duration","painAss","irritability"]
        # for i in ListChanges:
        #     req_info[i] = Dict_to_List(req_info[i])
        # print(req_info)

        PatientData.update_one(
            {"Patient_Id": SearchKey, "Assessment.Date": req_info['DateOfAssessment']},
            {"$set": {
                "Assessment.$.SeniorDoctorPrescription.FMSAssessment": 
                    req_info
                
            }}
        )

        #medKCO,personal,duration,painAss,irritability:

        return {"Status" : "Successful"}

@app.post("/TreatmentPrescription")
def TreatmentPrescription(info : dict):

    
    req_info = info
    req_info = dict(req_info)

    print(req_info)

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})

    if Find == None:
        return {"Status" : "Patient Not Found" }
    else:
        # Update the document in MongoDB:
        del req_info['Patient_Id']

        # ListChanges = ["medKCO","personal","duration","painAss","irritability"]
        # for i in ListChanges:
        #     req_info[i] = Dict_to_List(req_info[i])
        # print(req_info)

        status = PatientData.update_one(
            {"Patient_Id": SearchKey, "Assessment.Date": req_info['DateOfAssessment']},
            {"$set": {
                "Assessment.$.SeniorDoctorPrescription.TreatmentPrescription": req_info
            }}
        )
        EntityReVisit = {
            "Patient_Name" : Find['Patient_Name'],
            "Patient_Id" : Find['Patient_Id'],
            "ReviewDate" : req_info['reviewNext']            
        }

        ReVisit.insert_one(EntityReVisit)
        print("Status:",status.acknowledged)

        #medKCO,personal,duration,painAss,irritability:

        return {"Status" : "Successful"}

@app.get("/ReVisitPatients")
def ReVisitPatients():

    Result = ReVisit.find({'ReviewDate' : str(datetime.date.today())})
    Result = list(Result)
    for i in Result:
        del i['_id']
    return {"AllRevisit" : list(Result)}

@app.post("/GeneralAssessment")
def GeneralAssessment(info : dict):

    ## 
    req_info = info
    req_info = dict(req_info)

    ## print(req_info)

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found" }
    else:
        # Update the document in MongoDB:
        del req_info['Patient_Id']

        # ListChanges = ["medKCO","personal","duration","painAss","irritability"]

        # for i in ListChanges:
        #     req_info[i] = Dict_to_List(req_info[i])
        # ## print(req_info)

        Status = PatientData.update_one(
            {"Patient_Id": SearchKey, "Assessment.Date": req_info['DateOfAssessment']},
            {"$set": {
                "Assessment.$.SeniorDoctorPrescription": {
                    "GeneralAssessment": req_info,
                    "ShoulderAssessment": {},
                    "KneeAssessment": {},
                    "BalanceAssessment": {},
                    "LowBackAssessment": {},
                    "PARQPlusAssessment": {},
                    "FMSAssessment": {},
                    "TreatmentPrescription": {}
                }
            }}
        )

        print(Status)
        print(Status)

        

        #medKCO,personal,duration,painAss,irritability:

        return {"Status" : "Successful"}
    

@app.post("/GetGeneralAssessment")
def GetGeneralAssessment(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']

    Find = PatientData.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found" }
    else:
        Find = dict(Find)
        Assessment = Find['Assessment']
        try:
            for i in Assessment:
                if i['Date'] == req_info['Date']:
                    ResultSend = i['SeniorDoctorPrescription']['GeneralAssessment']
                    return ResultSend
        except:
            return {"Status" : "Not Found"}

## getting details
@app.post("/GetShoulderAssessment")
def GetShoulderAssessment(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']

    Find = PatientData.find_one({'Patient_Id' : SearchKey})

    try:
        if Find == None:
            return {"Status" : "Patient Not Found" }
        else:
            Find = dict(Find)
            Assessment = Find['Assessment']
            for i in Assessment:
                if i['Date'] == req_info['Date']:
                    ResultSend = i['SeniorDoctorPrescription']['ShoulderAssessment']
                    return ResultSend
    except:
        return {"Status" : "Not Found"}

@app.post("/GetKneeAssessment")
def GetKneeAssessment(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']
    print("trigger1")
    Find = PatientData.find_one({'Patient_Id' : SearchKey})

    try:
        if Find == None:
            return {"Status" : "Patient Not Found" }
        else:
            Find = dict(Find)
            Assessment = Find['Assessment']
    
            for i in Assessment:
                if i['Date'] == req_info['Date']:
                    ResultSend = i['SeniorDoctorPrescription']['KneeAssessment']
                    return ResultSend
    except:
        return {"Status" : "Not Found"} 

@app.post("/GetBalanceAssessment")
def GetBalanceAssessment(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']

    try:
        Find = PatientData.find_one({'Patient_Id' : SearchKey})
        if Find == None:
            return {"Status" : "Patient Not Found" }
        else:
            Find = dict(Find)
            Assessment = Find['Assessment']
            for i in Assessment:
                if i['Date'] == req_info['Date']:
                    ResultSend = i['SeniorDoctorPrescription']['BalanceAssessment']
                    return ResultSend
    except:
        return {"Status" : "Not Found"}
    
@app.post("/GetLowBackAssessment")
def GetLowBackAssessment(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']

    try:

        Find = PatientData.find_one({'Patient_Id' : SearchKey})
        if Find == None:
            return {"Status" : "Patient Not Found" }
        else:
            Find = dict(Find)
            Assessment = Find['Assessment']
            for i in Assessment:
                if i['Date'] == req_info['Date']:
                    ResultSend = i['SeniorDoctorPrescription']['LowBackAssessment']
                    return ResultSend
    except:
        return {"Status" : "Not Found"}
    
@app.post("/GetPARQPlusAssessment")
def GetPARQPlusAssessment(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']

    try:

        Find = PatientData.find_one({'Patient_Id' : SearchKey})
        if Find == None:
            return {"Status" : "Patient Not Found" }
        else:
            Find = dict(Find)
            Assessment = Find['Assessment']
            for i in Assessment:
                if i['Date'] == req_info['Date']:
                    ResultSend = i['SeniorDoctorPrescription']['PARQPlusAssessment']
                    return ResultSend
    except:
        return {"Status" : "Not Found"}
    
@app.post("/GetFMSAssessment")
def GetFMSAssessment(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']

    try:
        Find = PatientData.find_one({'Patient_Id' : SearchKey})
        if Find == None:
            return {"Status" : "Patient Not Found" }
        else:
            Find = dict(Find)
            Assessment = Find['Assessment']
            for i in Assessment:
                if i['Date'] == req_info['Date']:
                    ResultSend = i['SeniorDoctorPrescription']['FMSAssessment']
                    return ResultSend
    except:
        return {"Status" : "Not Found"}
    
@app.post("/GetTreatmentPrescription")
def GetTreatmentPrescription(info : dict):
    
    req_info = info
    req_info = dict(req_info)
    

    SearchKey = req_info['Patient_Id']

    try:
        Find = PatientData.find_one({'Patient_Id' : SearchKey})
        if Find == None:
            return {"Status" : "Patient Not Found" }
        else:
            Find = dict(Find)
            Assessment = Find['Assessment']
            for i in Assessment:
                if i['Date'] == req_info['Date']:
                    ResultSend = i['SeniorDoctorPrescription']['TreatmentPrescription']
                    return ResultSend
    except:
        return {"Status" : "Not Found"}
    
@app.post("/UpdateReview")
def UpdateReview(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']
    Find = ReviewData.find_one({'Patient_Id' : SearchKey , "DateOfReview" :  req_info['DateOfReview']})
    print()
    if Find == None:
        return {"Status" : "Patient Not Found" }
    
    SearchKey = req_info['Patient_Id']
    Find = ReviewData.find_one({'Patient_Id' : SearchKey})
    Find = list(Find)
    print(Find)

    if Find == []:
        return {"Status" : "Review not found"}
    else:
        Status = ReviewData.update_one(
            {"Patient_Id": SearchKey, "DateOfReview": req_info['DateOfReview']},
            {"$set": {
                "srDocNote": req_info['srDocNote'],
                "SeniorDoctorViewed" : True
                }
            }
        )
        return {"Status" : "Updated"}

@app.get("/AllReviews")
def AllReviews():
    Find = ReviewData.find({})
    Find = list(Find)

    FinalList = []

    for i in Find:
        # if i['SeniorDoctorViewed'] == False:
        del i['_id']
        FinalList.append(i)

    return {"AllReviews" : FinalList[::-1][:10]}

@app.get("/ReviewCount")
def ReviewCount():
    Find = ReviewData.find({'SeniorDoctorViewed' : False})
    Find = list(Find)
    FinalList = []
    for i in Find:
        if i['SeniorDoctorViewed'] == False:
            del i['_id']
            FinalList.append(i)

    return {"ReviewCount" : len(FinalList[::-1])}

@app.post("/ViewReview")
def ViewReview(info : dict):
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found" }
    
    SearchKey = req_info['Patient_Id']
    Find = ReviewData.find_one({'Patient_Id' : SearchKey , "DateOfReview" : req_info['DateOfReview']})

    if Find != None:
        del Find['_id']
        return Find
    else:
        return {"Status" : "Not Found"}



################# -------------------- Junior Route --------------------- #################

@app.post("/GetTreatmentTracker")
def TreatmentTracker(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})
    
    if Find == None:
        return {"Status" : "Patient Not Found" }
    else:
        ResultDict = {}
        Find = dict(Find)
        ResultDict['Patient_Id'] = Find['Patient_Id']
        ResultDict['Patient_Name'] = Find['Patient_Name']
        ResultDict['Patient_Age'] = Find['Patient_Age']
        ResultDict['Patient_Gender'] = Find['Patient_Gender']
        ResultDict['Patient_Height'] = Find['Patient_Height']
        ResultDict['Patient_Weight'] = Find['Patient_Weight']
        ResultDict['Patient_Contact_No'] = Find['Patient_Contact_No']
    
        ListOfItems = []

        for i in Find['Assessment']:
            CurrDict = {}
            CurrDict['GeneralAssessmentDate'] = i['Date']
            CurrDict['DateWise'] = i['JuniorDoctorPrescription']['DayWise']
            ListOfItems.append(CurrDict)
        
        ResultDict['DailyReview'] = ListOfItems[::-1]

        print(ListOfItems)

        return ResultDict

@app.post("/UpdateTreatmentTracker")
def TreatmentTracker(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})
    
    if Find == None:
        return {"Status" : "Patient Not Found" }
    
    updateDayWise = None

    for i in Find['Assessment']:
        if i['Date'] == req_info['GeneralAssessmentDate']:
            updateDayWise = i['JuniorDoctorPrescription']['DayWise']
            break
    
    RecievedDates = [i['Date'] for i in req_info['DateWise']]
    SetOfDates = set(RecievedDates)

    if len(SetOfDates) < len(RecievedDates):
        return {"status" : "duplicates exists"}

    updateDayWise = req_info['DateWise']
    
    Status = PatientData.update_one(
            {"Patient_Id": SearchKey, "Assessment.Date": req_info['GeneralAssessmentDate']},
            {"$set": {
                "Assessment.$.JuniorDoctorPrescription": {
                    "DayWise": updateDayWise
                }
            }}
        )
    
    if Status.acknowledged == True:
        return {"Status" : "Successful"}
    else:
        return {"Status" : "Not Successful"}

@app.post("/RaiseReview")
def RaiseReview(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})
    
    if Find == None:
        return {"Status" : "Patient Not Found" }
    
    SearchKey = req_info['Patient_Id']
    Find = ReviewData.find({'Patient_Id' : SearchKey})
    Find = list(Find)
    
    for i in Find:
        if i['DateOfReview'] == req_info['DateOfReview']:
            return {"Status" : "Review already exists for this date"}
    

    req_info['SeniorDoctorViewed'] =  False
    
    Check = ReviewData.insert_one(req_info)

    if Check.acknowledged == True:
        return {"Status" : "successful"}
    else:
        return {"Status" :  "not successful"}
    

##################### ------------ All Trainer / Re-Hab Routes ----------------------- ################

@app.post("/trainer/AddPatientBasic")
def AddPatientBasic(info : dict):
    
    req_info = info
    req_info = dict(req_info)


    print(req_info)

    SearchKey = req_info['Patient_Id']
    Find = PatientData.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found" }
    
    ReHabSearch = ReHab.find_one({'Patient_Id' : SearchKey})
    if ReHabSearch != None:
        if ReHabSearch['DateOfAssessment'] == req_info['DateOfAssessment']:
            return {"Status" : "Same date 2 visits"}
    
    req_info['ExerciseSchedule'] = []
    req_info['ExerciseTracking'] = []
    req_info['ScheduleDoneBy']= ""
    req_info['TrainerName']= ""
    
    Check = ReHab.insert_one(req_info)

    if Check.acknowledged == True:
        return {"Status" : "successful"}
    else:
        return {"Status" :  "not successful"}

@app.post("/trainer/PARQ_Assessment")
def PARQ_Assessment(info : dict):
    
    req_info = info
    req_info = dict(req_info)


    print(req_info)

    SearchKey = req_info['Patient_Id']
    Find = ReHab.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found in Re-Hab" }
    
    myquery = { "Patient_Id": SearchKey }
    del req_info['Patient_Id']
    newvalues = { "$set": { "PARQ_Assessment": req_info } }
    ReHab.update_one(myquery, newvalues)
    return {"Status" : "Successfully"}
      
@app.post("/trainer/AddExerciseSchedule")
def ExerciseSchedule(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']
    Find = ReHab.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found in Re-Hab" }
    
    myquery = { "Patient_Id": SearchKey }
    del req_info['Patient_Id']
    newvalues = { "$set": { "ExerciseSchedule": req_info['ExerciseSchedule'] , "ScheduleDoneBy" : req_info['ScheduleDoneBy']} }

    ReHab.update_many(myquery, newvalues)
    return {"Status" : "Successfully"}

@app.post("/trainer/ViewExerciseSchedule")
def ViewExerciseSchedule(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']
    Find = ReHab.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found in Re-Hab" }
    else:
        Find = dict(Find)
        return Find['ExerciseSchedule']
    
@app.post("/trainer/AddExerciseTracking")
def ExerciseTracking(info : dict):

    
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']
    Find = ReHab.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found in Re-Hab" }
    
    myquery = { "Patient_Id": SearchKey }
    del req_info['Patient_Id']
    newvalues = { "$set": { "ExerciseTracking": req_info['ExerciseTracking'] , "TrainerName" : req_info['TrainerName'] } }
    ReHab.update_many(myquery, newvalues)
    return {"Status" : "Successfully"}

@app.post("/trainer/ViewExerciseTracking")
def ViewExerciseTracking(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']
    Find = ReHab.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found in Re-Hab" }
    else:
        Find = dict(Find)
        return Find['ExerciseTracking']
    
@app.post("/trainer/ViewRehabView")
def ViewRehabView(info : dict):
    
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']
    print(SearchKey)
    Find = ReHab.find_one({'Patient_Id' : SearchKey})
    if Find == None:
        return {"Status" : "Patient Not Found in Re-Hab" }
    else:
        Find = dict(Find)
        del Find['_id']
        return convert_to_second_json_format(Find)
    

#################  -------------------------------- all app related routes --------------------------- ################

@app.post("/app/login")
def app_login(info : dict):
    req_info = info
    userId = req_info['userId']
    password = req_info['password']
    result = PatientData.find_one({"Patient_Id" : userId})

    # Check if the entity exists or not
    if result and password == userId:
        return True
    else:
        return False


@app.post("/app/ViewPatientData")
def ViewPatientData(info : dict):
    req_info = info
    req_info = dict(req_info)

    SearchKey = req_info['Patient_Id']
    
    query = {"Patient_Id": SearchKey}
    projection = {
        "Patient_Id": 1,
        "Patient_Name": 1,
        "Patient_Age": 1,
        "Patient_Gender": 1,
        "Patient_Height": 1,
        "Patient_Weight": 1,
        "Patient_Contact_No": 1,
        "Occupation": 1,
        "Address": 1,
        "Assessment": {
            "$elemMatch": {
                "SeniorDoctorPrescription.TreatmentPrescription.DateOfAssessment": {
                    "$exists": True
                }
            }
        }
    }

    # Sort the assessments in descending order based on the date
    sort_order = [("Assessment.SeniorDoctorPrescription.TreatmentPrescription.DateOfAssessment", pymongo.DESCENDING)]

    # Execute the query and get the result
    result = PatientData.find_one(query, projection, sort=sort_order)

    if result:
        # Extract the required details from the result
        patient_details = {
            "Patient_Id": result["Patient_Id"],
            "Patient_Name": result["Patient_Name"],
            "Patient_Age": result["Patient_Age"],
            "Patient_Gender": result["Patient_Gender"],
            "Patient_Height": result["Patient_Height"],
            "Patient_Weight": result["Patient_Weight"],
            "Patient_Contact_No": result["Patient_Contact_No"],
            "Occupation": result["Occupation"],
            "Address": result["Address"]
        }

        # Extract the latest assessment details
        assessment = result["Assessment"][0]
        latest_assessment = {
            "Diagnosis": assessment["SeniorDoctorPrescription"]["TreatmentPrescription"]["diagnosis"],
            "DateOfAssessment": assessment["SeniorDoctorPrescription"]["TreatmentPrescription"]["DateOfAssessment"],
            "Exercise": assessment["SeniorDoctorPrescription"]["TreatmentPrescription"]["exercises"],
            "TreatmentPlan": assessment["SeniorDoctorPrescription"]["TreatmentPrescription"]["treatmentPlan"],
            "NumberOfDays" : assessment["SeniorDoctorPrescription"]["TreatmentPrescription"]["numberOfDays"],
            "HomeAdvice" : assessment["SeniorDoctorPrescription"]["TreatmentPrescription"]["homeAdvice"],
            "ReviewNext" : assessment["SeniorDoctorPrescription"]["TreatmentPrescription"]["reviewNext"],
            "FollowUp" : assessment["SeniorDoctorPrescription"]["TreatmentPrescription"]["followUp"],
            "Contraindication" : assessment["SeniorDoctorPrescription"]["TreatmentPrescription"]["contraindication"],
            "DayWise" : assessment["JuniorDoctorPrescription"]["DayWise"]
        }

        print(assessment)

        

        # Add links to exercise data
        for exercise_entry in latest_assessment['Exercise']:
            exercise_name = exercise_entry['NameOfExercise']
            
            for exercise_category, exercises in exercise_links.items():
                for exercise in exercises:
                    if exercise['Title'] == exercise_name:
                        exercise_entry['Link'] = exercise['Link']
                        break

        return {"patient_details" : patient_details, "latest_assessment" : latest_assessment}
    else:
        return None





################ ------------------- End of all the routes ------------------ #####################


