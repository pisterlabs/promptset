import json, time

import openai

openai.api_key_path = "openai.key"

def getEmbedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

with open("f23.json", "r") as f:
    semData = json.load(f)

allCourseData = semData["courses"]
courseDescriptionEmbeddings = {}

try:
    seenLast = True
    for courseNum, courseData in allCourseData.items():
        # Filter out some courses for now!
        #if courseNum[:2] not in ["15", "16", "08", "02", "11", "05", "10", "17", "18", "60", "54"]:
        if courseNum[:2] not in ["33", "76", "98", "03", "70", "09"]:
            continue

        if courseNum == "XX-XXX":
            seenLast = True

        if not seenLast:
            continue

        print(courseNum)
        description = courseData["desc"]

        if description is None:
            continue

        descEmbedding = getEmbedding(description)
        time.sleep(1.5)

        courseDescriptionEmbeddings[courseNum] = descEmbedding
except Exception as e:
    print(e)

with open("f23embeddings.json", "w+") as f:
    json.dump(courseDescriptionEmbeddings, f)    
