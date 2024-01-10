# IF CONFIGURED WITH API KEYS, RUNNING THIS SCRIPT WILL COST MONEY
# This script retrieves courses from the PeterPortal API, uses OpenAI's api to generate embeddings, 
# and outputs it to a JSON file

import requests
import openai
import os
import json
from dotenv import load_dotenv
import time

def createCourseList() -> [dict()]:
    try:
        response = requests.get("https://api.peterportal.org/rest/v0/courses/all")
    except:
        raise RuntimeError("Error while requesting API")
    courseList = []
    for i, courseObject in enumerate(response.json()):
        print(f'{i} | {courseObject["id"]}')
        if (i % 1500 == 0 and i != 0): #
            print('WAITING... (so we stay under OpenAIs 3,000 RPM limit)')
            time.sleep(30)
        try:
            courseDict = {
                "id": courseObject["id"],
                "title": courseObject["title"],
                "department": courseObject["department"],
                "description": courseObject["description"],
                "embedding": _getEmbedding(f'{courseObject["title"]}. {courseObject["description"]}')
            }
            courseList.append(courseDict)
        except KeyError:
            print(f'KEYERROR for {courseObject["id"]}')
            pass

    return courseList

def createCourseListPinecone(): #Creates a JSON in the format Pinecone requires
    try:
        response = requests.get("https://api.peterportal.org/rest/v0/courses/all")
    except:
        raise RuntimeError("Error while requesting API")
    
    courseList = {}
    courseList["vectors"] = []
    for i, courseObject in enumerate(response.json()):
        print(f'{i} | {courseObject["id"]}')
        if (i % 3000 == 0 and i != 0): #
            print('WAITING... (so we stay under OpenAIs 3,000 RPM limit)')
            time.sleep(30)
        try:
            courseDict = {}
            courseDict["id"] = courseObject["id"]
            print(courseDict.keys())
            courseDict["metadata"] = {
                "title": courseObject["title"],
                "department": courseObject["department"],
                "description": courseObject["description"]
            }
            courseDict["values"] = _getEmbedding(f'{courseObject["title"]}. {courseObject["description"]}')
            
            courseList["vectors"].append(courseDict)
        except KeyError:
            print(f'KEYERROR for {courseObject["id"]}')
            pass

    return courseList

def _getEmbedding(text, model="text-embedding-ada-002") -> [float]:
   try:
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
   except:
       raise RuntimeError("getEmbedding() error")
       

def createPineconeJsonFile(courseList, fileName="../data/coursesWithEmbeddingsForPinecone.json"):
    with open(fileName, "w") as outfile:
        json.dump(courseList, outfile)

if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    courseList = createCourseListPinecone()
    createPineconeJsonFile(courseList)
  
    # s3 = getS3Bucket()
    # for bucket in s3.buckets.all():
    #     print(bucket.name)
    
