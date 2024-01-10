import json
import openAI_settings as ai
from pymongo import MongoClient
import os
import pandas as pd
import FormatFile as ff
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


#Function that directly inputs into the OpenAI.
def aiInput(reviewStr,app_name,id,reviewNum):
	
	print(f"Scoring review No.{reviewNum} for app {app_name}")

	#Prompt moved to external python file to promote modularity and ease of access for updating.
	prompt = ff.retPrompt(reviewStr)

	response=ai.get_completion(prompt)
	print("Scoring complete.\n")
	print("Formatting the response as a json file.")
	print(response)
	#Calls upon function that formats OpenAI output as JSON and fills in empty fields
	#that were not relevant to the prompt to optimize token output.
	response_json= jsonFormatFillIn(response,reviewStr,id,app_name)
	return response_json

#Fills in missing information in the jsonFormat.
def jsonFormatFillIn(response,reviewStr,id,app_name):
	id = str(id)[10:-2]
	response_json = json.loads(response)
	#Adding in data that was irrelevant to the prompt to optimize output
	response_json['Content'] = reviewStr
	response_json['ID'] = id
	response_json['App Name'] = app_name
	return response_json


#Reads excel doc using Pandas library and returns them to main function
def appNameListGenerate():
	#Reads excel doc using Pandas library
	print("Creating app name list")
	excelFilePath=os.getenv("EXCELFILEPATH")
	ios_excel=pd.read_excel(excelFilePath,usecols='A')
	return list(ios_excel['App-Name'])

#Stores what app was last passed through Open AI
def appNameStoreWrite(num,SINGLEINPUT,filepath):
	if (SINGLEINPUT!=True):
		filepath+="OpenAiAppNum.txt"
		appNameStore=open(filepath,"w")
		appNameStore.write(str(num))
		appNameStore.close()

#Stores what app review the openAI prompt last left off on
def reviewStoreWrite(num,SINGLEINPUT,filepath):
	filepath+="OpenAIReviewNum.txt"
	if (SINGLEINPUT!=True):
		reviewStore=open(filepath,"w")
		reviewStore.write(str(num))
		reviewStore.close()
		print(f"Review Storage File Incremented: {num}")

#Returns what app # the AI review process left off on.
def appStoreRead(SINGLEINPUT,filepath):
	if SINGLEINPUT==False:
		filepath+="OpenAIAppNum.txt"
		try:
			appNameStore= open(filepath,"r")
			testNum=appNameStore.readline()
			appNameStore.close()
			if testNum.isnumeric():
				return int(testNum)
		except FileNotFoundError:
			pass
	print("Last app not found")
	return 0

#Returns what app # the AI review process left off on.
def reviewStoreRead(SINGLEINPUT,filepath,RESET):
	retVal=0
	filepath+="OpenAiReviewNum.txt"
	if (SINGLEINPUT!=True and RESET==False):
		try:
			reviewStore=open(filepath,"r")
			testNum=reviewStore.readline()
			if testNum.isnumeric():
				return int(testNum)
			reviewStore.close()
		except FileNotFoundError:
			pass
	print("File not found for review store num")
	return retVal

#Generates the file name of the output json file.
def fileGeneration(response_json,reviewNum,app_name,filepath):

	#The index of the first non-alphanumeric character in an app name.
	nonAlNumIndex=-1
	for y in range(0,len(app_name)):
		if (app_name[y].isalnum()):
			pass
		else:
			nonAlNumIndex=y
			break
	if filepath=="./Generated Files/Storage_BR/":
		filepath="./Generated Files/Reviews_BR/"
	else:
		filepath="./Generated Files/Reviews_AP"
	if nonAlNumIndex!=-1:
		#Stops app name at non-alphanumeric character to ensure file name is possible
		app_name=app_name[0:int(nonAlNumIndex)]
	filename=f"{app_name}_review_{reviewNum}"
	try:
		with open(filepath+filename+".json", 'r') as demoFile:
			demoFile.readline
		filename+="_2.json"
	except FileNotFoundError:
		filename+=".json"
	filepath+=filename
	# Write the OpenAI output to a JSON file
	with open(filepath, 'w') as json_file:
		json.dump(response_json, json_file, indent=4)

	print("Completed!\n")
	print("Response:")
	print(f"{response_json}\n")
