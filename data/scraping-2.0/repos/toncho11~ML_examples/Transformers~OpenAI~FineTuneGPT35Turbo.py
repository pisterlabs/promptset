# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 09:46:53 2023

An example on how to use gpt-3.5-turbo for fine-tuning (training).
The code handles progress and errors during the fine-tuning process.

pip install openai

It uses dummy training and test data.

"""

import json
import openai
import time
import subprocess
import json

#api_key ="YOUR_OPENAI_API_KEY"
keyFile = open('OpenAiApiKey.txt', 'r')
api_key = keyFile.readline()
openai.api_key = api_key

file_name = "training_data.json"

#1. First create a json training file 

data = {"messages": [
            {"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."},
            {"role": "user", "content": "What's the capital of France?"},
            {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}
       ]}

json_dump = json.dumps(data)

# Writing to sample.json
with open(file_name, "w") as outfile:
    #create 10 test samples
    for i in range(0,10):
        outfile.write(json_dump + "\n")
  
#end of generating json file

#2. Upload training data to OpenAI
upload_response = openai.File.create(
  file=open(file_name, "rb"),
  purpose='fine-tune'
)
file_id = upload_response.id

#3. Fine-tune model
fine_tune_response = openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo")

print("Started Training ...")
job_id = fine_tune_response.id

error = False
done =  False

response = openai.FineTuningJob.list_events(id=job_id, limit=50)

# while True:
#     res = openai.FineTuningJob.retrieve(job_id)
#     if res["finished_at"] != None:
#         break
#     else:
#         print(".", end="")
#         sleep(100)
        
while (True): #it prints some of the old messages as well
   
    events = response["data"]
    events.reverse()
    
    for event in events:
        print(event["message"])
        if event["data"] != None and event["data"] != {}:
            if "error_code" in event["data"]:
                error = True
                print("Error code:",  event["data"]["error_code"])
                break
        if "The job has successfully completed" in event["message"]:
            done = True
            break
    
    if error or done:
        break
    
    time.sleep(10)
    response = openai.FineTuningJob.list_events(id=job_id, limit=50)
        
if error:
    print("Training has completed with error!")
    raise SystemExit()
elif done:
    print("Training completed successfully.")

print("##############################################################")

print("Testing the newly trained model ...")

#4. Get the final trained model
response = openai.FineTuningJob.retrieve(fine_tune_response.id)
fine_tuned_model = response["fine_tuned_model"]

if fine_tuned_model == None:
    print("Model is empty!")
    raise SystemExit()

#5. Testing the newly fine-tuned model

completion = openai.ChatCompletion.create(
  model = fine_tuned_model,
  messages=[
    {"role": "system", "content": "You are Zordon, leader of the Power Rangers."},
    {"role": "user", "content": "Zordon, the Red Ranger has been captured! What do we do?"}
  ]
)
print(completion.choices[0].message)