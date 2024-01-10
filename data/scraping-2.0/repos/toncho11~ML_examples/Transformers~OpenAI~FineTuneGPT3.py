# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 09:46:53 2023

Original article: https://www.indiehackers.com/post/how-to-fine-tune-a-gpt-3-model-using-python-with-your-own-data-for-improved-performance-198dfe51d6

This script shows how to fine-tune a GPT-3 model using Python with your 
own data for improved performance.
The fine=tuning is not done locally on your computer, but on the remote servers of OpenAI
You will need an API access key.
Extra code has been added that waits until the model is trained (or has failed)

The data need to be a JSONL document with a new prompt and the ideal generated text:

{"prompt": "<question>", "completion": "<ideal answer>"}
{"prompt": "<question>", "completion": "<ideal answer>"}
{"prompt": "<question>", "completion": "<ideal answer>"}

https://platform.openai.com/docs/models/gpt-3
Instruct models are optimized to follow single-turn instructions. Ada is the fastest model, 
while Davinci is the most powerful.

pip install openai

The code in this script waits until the fine-tuning is done by checking on the OpenAI web-site periodically.

The example here teaches the bot its name and age.

"""

import json
import openai
import time
import subprocess

#api_key ="YOUR_OPENAI_API_KEY"
keyFile = open('OpenAiApiKey.txt', 'r')
api_key = keyFile.readline()
openai.api_key = api_key

#1. First create a json training file 

# training_data = [{
#     "prompt": "Where is the billing ->",
#     "completion": " You find the billing in the left-hand side menu.\n"
# },{
#     "prompt":"How do I upgrade my account ->",
#     "completion": " Visit you user settings in the left-hand side menu, then click 'upgrade account' button at the top.\n"
# }]
training_data = [{
    "prompt": "What is your name? ->",
    "completion": " My name is John.\n"
   },{
    "prompt":"How old are you? ->",
    "completion": " I am thirty years old.\n"
    }
  ,{
     "prompt":"What is your age? ->",
     "completion": " My age is thirty years old.\n"}
  ,{
     "prompt":"What do you do for living? ->",
     "completion": " I work as an engineer.\n"}
  ,{
     "prompt":"What is your occupation? ->",
     "completion": " I work in the domain of engineering.\n"}
  ,{
     "prompt":"What is your profession? ->",
     "completion": " I am an engineer.\n"}
  ,{
     "prompt":"What do you do? ->",
     "completion": " I work as an engineer.\n"}
  ]
#Make sure to end each prompt with a suffix. According to the OpenAI API reference, you can use ->.
#Make sure to end each completion with a suffix as well -  for example '\n'.

#convert training data to JSON Lines file
file_name = "training_data.jsonl"

with open(file_name, "w") as output_file:
 for entry in training_data:
  json.dump(entry, output_file)
  output_file.write("\n")
  
#end of generating json file

#verify the json file in your console:
#openai tools fine_tunes.prepare_data -f training_data.jsonl

#2. Upload training data to OpenAI
upload_response = openai.File.create(
  file=open(file_name, "rb"),
  purpose='fine-tune'
)
file_id = upload_response.id

#3. Fine-tune model
fine_tune_response = openai.FineTune.create(training_file=file_id)

#The default model is "curie". But if you'd like to use DaVinci instead, then add it as a base model to fine-tune like this:
#openai.FineTune.create(training_file=file_id, model="davinci")

#use response.events to check progress
#latest event is: fine_tune_response.events[-1]

print("Started Training ...")
id = fine_tune_response.id
error = False
while True:
    cmd = "openai -k " + api_key + " api fine_tunes.get -i " + id
    result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    if "\"status\": \"succeeded\"" in str(result.stdout):
        print("Succeeded")
        break
    if result.stderr !=None:
        print("Error in cmd!")
        error = True
        break
    if "\"status\": \"failed\"" in str(result.stdout):
        print("Failed remote job!")
        error = True
        break
    time.sleep(20)
    print("Still training ...")
print("Done Training")

#4. Test the newly trained model
if error == False:
    
    retrieve_response = openai.FineTune.retrieve(id=fine_tune_response.id)
    
    #get the final trained model
    if fine_tune_response.fine_tuned_model != None:
        fine_tuned_model = fine_tune_response.fine_tuned_model
    else:
        fine_tuned_model = openai.FineTune.retrieve(id=fine_tune_response.id).fine_tuned_model
    
    if fine_tuned_model == None:
        print("Model is empty!")
    else:
        #Testing
        
        #Next, use the completion with the fine-tuned model previously created:
            
        #Remember to end the prompt with the same suffix as we used in the training data; ->:
        
        print("Results: ========================================================")
        prompts = ["What is your name? ->", "How old are you? ->", "What is your job? ->"]
        
        for p in prompts:
            answer = openai.Completion.create(
              model=fine_tuned_model,
              prompt=p,
              max_tokens=100,
              temperature=0, #lower value will make it more focused and deterministic
              n = 1
            )
            
            print("Question:", p)
            print("Answer:", answer['choices'][0]['text'])
            #print("Answer:", answer['choices'][0]['text'][0:answer['choices'][0]['text'].find(".")+1])
            
            print("-------------------------------------")
    
else:
    print("Error training model.")