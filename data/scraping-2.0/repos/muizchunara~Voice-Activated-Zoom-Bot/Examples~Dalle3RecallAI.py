from flask import Flask, request
from openai import OpenAI
from time import sleep
import threading
import requests
import json

#Needed API Keys
RECALL_KEY="YOUR_RECALLAI_API_KEY_HERE"
OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
trigger="imagine"

#Creating the OpenAI Client we will use to call DALLE-3
OpenAIClient = OpenAI(api_key=OPENAI_API_KEY)

#setting a baseline timestamp to make sure only recent transcript entries are checked
timestamp=0

#Creating an end flag to terminate the bot when the call ends (All bots run on their own thread)
endFlag=threading.Event()

#Creating the Recall.AI Zoom bot that will be transcribing the call, listening for voice triggers, and sending the generated image into the call
def createBot(meetingLink:str, name:str="Imagine"):
  
  createAPIUrl="https://api.recall.ai/api/v1/bot"
  headers={"Authorization":"Token "+RECALL_KEY,
           "Content-Type":"application/json"}
  
  #Setting the bot's display name and telling the bot which call it should be joining
  body={"meeting_url":meetingLink,
        "bot_name":name,
        "real_time_transcription": {"destination_url": "https://play.svix.com/in/e_8VpfB9IR4rjM9AadoR1fwMRoRJD/"},
        "transcription_options": {"provider": "gladia"}}
  
  #Post request to Recall.AI that creates the bot
  response=requests.post(createAPIUrl,data=json.dumps(body),headers=headers)
  
  return response


#A Basic function that uses the botID to get the call transcript (Live transcript if the call is ongoing)
#This transcript will then be used to check for the voice command, and assemble the image generation prompt
def getTranscript(botID:str):
  
  headers={"Authorization":"Token "+RECALL_KEY,
           "Content-Type":"application/json"}
  transcriptAPIUrl="https://api.recall.ai/api/v1/bot/"+botID+"/transcript/"
  
  #Basic REST Get request to pull the transcript
  response=requests.get(transcriptAPIUrl,headers=headers)
  return response

#Checking the transcript for a specific keyword trigger
def checkTranscript(data:[],id):
  
  #timestamp is in global scope to make sure it is carried over through checks
  global timestamp
  
  #converting the transcript data into a list of json objects
  data=data.json()
  
  #creating an empty to prompt to use to generate the image
  prompt=""
  
  #A basic check that uses the falsy nature of lists to make sure the transcript data is not empty
  if data:
    splitData=[]
    
    #splitting the transcript by speaker to make sure the bot is usable by all participants in the call
    for datapoint in data:
      splitData.append(datapoint)
    dataLen=len(data)
    
    #iterating through each speakers transcript starting with the most recent words, scanning for the trigger keyword "imagine"
    for speaker in splitData:
      dataLen=len(speaker['words'])  
      for i in range(dataLen-1,0,-1):
        
        #Making sure only words spoken after the last check are scanned
        if speaker['words'][i]['start_timestamp'] < timestamp:
          break
        
        #Once the trigger word is found, pull all the words spoken in the same sentence after the trigger word to assemble the image generation prompt
        if speaker['words'][i]['text'].lower() == trigger:
          prompt=extractPrompt(speaker,i,id)
          break
    
    #resetting the timestamp to prevent future checks from repeating data
      if(speaker['words'][dataLen-1]['start_timestamp']>timestamp):
        timestamp=speaker['words'][dataLen-1]['start_timestamp']
  
  return prompt

#extracting and assembling a prompt following keyword detection  
def extractPrompt(data,i,id):
  #punctuation used to detect the end of a sentence
  PUNCTUATION=".?!"
  
  #Creating an empty prompt to build the prompt off
  prompt=""
  
  #a flag detecting once punctuation has been found to end the prompt assembly
  punctFound=False
  dataLen=len(data['words'])
  
  #A loop that will iterate through the transcript with everything spoken after the trigger word to assemble a prompt, ending after a full sentence prompt has been assembled
  while((not punctFound)):
    
    #If the end of the sentence has not been spoken yet, the bot will wait 2 seconds before getting a fresh transcript and checking if the sentence has ended
    if(i >= dataLen):
      sleep(2)
      transcript=getTranscript(id).json()
      for datapoint in transcript:
        if(datapoint['speaker']==data['speaker']):
          data['words']=(datapoint['words'])
          dataLen=len(data['words'])
          
    #Checking if the currently iterated word contains the end of the sentence
    if(any(i in PUNCTUATION for i in data['words'][i]['text'])):
      punctFound=True
    
    #if the sentence has not ended, append the word in the transcript to the prompt
    if(i< dataLen):
      prompt+=data['words'][i]['text']
      i+=1      
    
  return prompt

#Using the prompt extracted from the transcript to generate an image using DALLE-3 and return a direct link to the image
def generateImage(prompt:str):
  
  #Creating a standard quality and size image request using the extracted prompt
  response = OpenAIClient.images.generate(
    model="dall-e-3",
    prompt=prompt,
    size="1024x1024",
    quality="standard",
    n=1,
  )

  #pulling the embed URL from the response that will be sent into Zoom
  image_url = response.data[0].url
  
  return image_url


#Sending the URL to the generated image into the zoom chat
def sendMessage(id, imageUrl):
  
  messageAPIUrl="https://api.recall.ai/api/v1/bot/"+id+"/send_chat_message/"
  
  body = { "message": imageUrl }
  headers = {
    "content-type": "application/json",
    "Authorization": "Token " + RECALL_KEY
  }
  
  response=requests.post(messageAPIUrl,data=json.dumps(body),headers=headers)
  return response
  
#A Loop that runs the bot through the call, periodically checking for a live transcript and trigger word
def runBot(id):
  while(True):
    #Conditional Logic to end the task and thread once the bot leaves the call using event flags
    if(endFlag.is_set()):
      break
    
    #pulling and checking the transcript
    transcript=getTranscript(id)
    prompt=checkTranscript(transcript,id)
    
    #if a valid prompt is returned, an image will be generated then sent into the zoom call
    if prompt:
      imageUrl = generateImage(prompt)
      sendMessage(id,imageUrl)
    
    #Adding a delay to the bot running to keep API usage costs low, and prevent excessive network usage
    sleep(5)
  return

app = Flask(__name__)

#The bots all run via webhooks intended to minimize the need for manual intervention
@app.route("/webhook", methods=["POST"])
def webhook_listener():
    data = request.get_json()
    
    #If a create bot webhook is recieved (Webhook Structure: {"create-bot: MEETING_URL"}), a bot will be created and sent into the specified meeting via invite link
    if "create_bot" in data:
      createBot(data["create_bot"])
    
    #Checking if the incoming webhook is a Recall.AI event webhook
    if "event" in data:
      #If the incoming webhook is a bot indicating it has entered a call and is ready to record, create a thread to run the keyword extraction and image generation
      if(data["data"]["status"]["code"]=="in_call_recording"):
        t1=threading.Thread(target=runBot,args=(data["data"]["bot_id"],))
        t1.daemon=True #Running the thread as a daemon to prevent memory leaks should the main service crash
        t1.start()
        
      #If the incoming webhook indicates the bot has completed its for any reason such as the call ending, being removed from the call, etc., send a flag to the associated service to terminate the task
      elif(data["data"]["status"]["code"]=="call_ended" or data["data"]["status"]["code"]=="done"):
        endFlag.set()
    
    #provide a basic webhook response on webhook receipt
    return "Webhook received successfully", 200

#launch the flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)