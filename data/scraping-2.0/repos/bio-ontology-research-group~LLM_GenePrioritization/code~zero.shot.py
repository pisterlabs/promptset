#This code interacts with openAI's API to use GPT with zero-shot prompting.
#run this code as: python zero.shot.py prompt_index start_patient end_patient
#example:  python zero.shot.py 0 1 101

import openai
import os
import pandas as pd
import time
import sys

openai.api_key = 'your_openAI_API_key'

def get_completion(prompt, model="gpt-4"): 

	messages = [{"role": "user", "content": prompt}]
	try:
		response = openai.ChatCompletion.create(

		model=model,

		messages=messages,

		temperature=0,)

		return response.choices[0].message["content"]
	except openai.error.OpenAIError as e:
		print ("OpenAI API error:", e)
		return get_completion(prompt, model)



#-------main-----------------------
for i in range(int(sys.argv[2]),int(sys.argv[3])):
	print ("Patient:"+str(i))
	f=open ("./questions/q_"+str(i)+".txt","r")
	prompt=f.readlines()[int(sys.argv[1])] #0,1,2, #line number of the file
	f.close()
	response = get_completion(prompt)
	if response is not None:
		if "AI language model" in response or "not possible" in response.lower() or "1. " not in response.lower():
			prompt=prompt.rstrip()+". In the case of not enough information, still try to rank these genes by using gene function, site of expression, or information from animal models."
			response = get_completion(prompt)
		file = open("./responses/gene"+str(int(sys.argv[1])*5+5)+"/r_"+str(i)+".txt", "w")
		file.write(response)
		file.close()
	else:
		print ("Error")

