# 1. 필요한 라이브러리 임포트
import requests
import json
import openai
from dotenv import load_dotenv
import os
import time

# # 1. check_bio
url = "http://localhost:8010/check_bio/"
# data = {"linkedin_url": "https://www.linkedin.com/in/jongwon-park-247692147/"}
data = {"linkedin_url": "https://www.linkedin.com/in/soomin-hwang-72b635254/"}
start_time = time.time()
res = requests.post(url, data=json.dumps(data), timeout=600)
end_time = time.time()
res = json.loads(res.text)
print(res)
print(f"elapsed time: {end_time - start_time} seconds")

# # 2. generate_initial_ideas
url = "http://localhost:8010/generate_initial_ideas/"
data = {"linkedin_summary": res["linkedin_summary"], "interests": "software, ai, medical"}
res = requests.post(url, data=json.dumps(data))
res_json = json.loads(res.text)
print(res_json)

# 3. discuss to business plan
url = "http://localhost:8010/discuss/"

data = {
    "log": {
        "service_name" : res_json["ideas"][0]["service_name"],
        "problem": res_json["ideas"][0]["problem"],
        "service_idea" : res_json["ideas"][0]["service_idea"],
        "dialog" : []
    },
    "human_input": "",
    "speaker_list": [],
}
res = requests.post(url, data=json.dumps(data))

message = json.loads(res.text)
dialog = [{f"{message['speaker']}": f"{message['contents']}\n"}]
print(dialog)
while not message["is_finished"]:
	data = {
		"log": {
			"service_name" : res_json["ideas"][0]["service_name"],
			"problem": res_json["ideas"][0]["problem"],
			"service_idea" : res_json["ideas"][0]["service_idea"],
			"dialog" : dialog
		},
		"human_input": "",
		"speaker_list": [],
	}
	res = requests.post(url, data=json.dumps(data))
	message = json.loads(res.text)
	dialog.append({f"{message['speaker']}": f"{message['contents']}\n"})
	print(dialog)

data = {
		"log": {
			"service_name" : res_json["ideas"][0]["service_name"],
			"problem": res_json["ideas"][0]["problem"],
			"service_idea" : res_json["ideas"][0]["service_idea"],
			"dialog" : dialog
		},
		# "human_input": "ok. continue discussion.",
		# "speaker_list": [2, 1], # Test human order
		"human_input": "",
		"speaker_list": [], # Test autogpt
	}
res = requests.post(url, data=json.dumps(data))
message = json.loads(res.text)

dialog.append({f"{message['speaker']}": f"{message['contents']}\n"})
print(dialog)

while not message["is_finished"]:
	data = {
		"log": {
			"service_name" : res_json["ideas"][0]["service_name"],
			"problem": res_json["ideas"][0]["problem"],
			"service_idea" : res_json["ideas"][0]["service_idea"],
			"dialog" : dialog
		},
		"human_input": "",
		"speaker_list": [],
	}
	res = requests.post(url, data=json.dumps(data))
	message = json.loads(res.text)
	dialog.append({f"{message['speaker']}": f"{message['contents']}\n"})
	print(dialog)

url = "http://localhost:8010/generate_business_plan/"
data = {
		"log": {
			"service_name" : res_json["ideas"][0]["service_name"],
			"problem": res_json["ideas"][0]["problem"],
			"service_idea" : res_json["ideas"][0]["service_idea"],
			"dialog" : dialog
		}
}
res = requests.post(url, data=json.dumps(data))
bp = json.loads(res.text)
print(bp)




