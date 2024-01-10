# import requests
# import json
import time
import openai
import sys
sys.path.append('/Users/busterblackledge/')
from keys import openai_API_key

def get_chat_completion():
	openai.api_key = openai_API_key
	#url = "https://api.openai.com/v1/chat/completions"
	#headers = {
	#	"Authorization": "Bearer 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",
	#	"Content-Type": "application/json"
	#}
	#data = {
	#	"model": "gpt-3.5-turbo",
	#	"messages": [{"role": "user", "content": f"explain the following point in a single paragraph in the style of an excited lecturer: {line}"}]
	#}
	#response = requests.post(url, headers=headers, data=json.dumps(data))
	#return response.json()    
	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		messages=[
				{"role": "system", "content": "How are stars born"},
			]
	)	
	return response

num_iters = 2
sum = 0
for i in range(num_iters):
	start_time = time.time()
	response = get_chat_completion()
	end_time = time.time()
	sum += end_time - start_time

av_time = sum/num_iters 

print(response['choices'][0]['message']['content'])
print(f"Average execution time: {av_time}")
