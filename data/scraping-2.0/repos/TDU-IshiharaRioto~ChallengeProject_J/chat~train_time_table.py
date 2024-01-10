import openai
import os

openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_version = "2023-07-01-preview"
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")


messages = [
    {"role": "user", "content": "運行情報を教えてください"}
]
	

functions = [
	{
		"name":"teach_train_time_table",
		"description": "電車の運行情報を教えてください",
		"parameters":{
			"type": "object",
                "properties": {
					"line name": {
                        "type": "string",
                        "description": "聞かれている路線"
					}
			},
			"requaired":[]
		}

	},
	
]

response = openai.ChatCompletion.create(
    engine="chat",#デプロイ名
    messages=messages,
    functions=functions,
    function_call="auto",
)
response_message = response["choices"][0]["message"]

print(response_message["function_call"]["arguments"])

